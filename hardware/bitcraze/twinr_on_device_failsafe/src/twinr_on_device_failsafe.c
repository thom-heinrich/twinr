/**
 * Twinr on-device failsafe app.
 *
 * This Crazyflie app-layer module keeps one tiny deterministic safety loop on
 * the STM32. While the host sends a fresh Appchannel heartbeat the app stays in
 * monitoring mode. If the heartbeat goes stale, the battery sags too far, or a
 * close obstacle is detected, the app takes over locally and executes a small
 * reactive brake-and-descend landing sequence. The local controller keeps
 * sending setpoints until touchdown is confirmed; it does not cut motors in
 * mid-air.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "app.h"
#include "app_channel.h"
#include "commander.h"
#include "debug.h"
#include "FreeRTOS.h"
#include "log.h"
#include "param.h"
#include "supervisor.h"
#include "task.h"

#define DEBUG_MODULE "TWINRFS"

#define TWINR_FS_PROTOCOL_VERSION 1
#define TWINR_FS_PACKET_KIND_HEARTBEAT 1
#define TWINR_FS_PACKET_KIND_STATUS 2
#define TWINR_FS_FLAG_ENABLE (1U << 0)
#define TWINR_FS_FLAG_REQUIRE_CLEARANCE (1U << 1)

#define TWINR_FS_LOOP_PERIOD_MS 20
#define TWINR_FS_STATUS_PERIOD_MS 100
#define TWINR_FS_HEARTBEAT_TIMEOUT_MS 350
#define TWINR_FS_LOW_BATTERY_MV 3550
#define TWINR_FS_CRITICAL_BATTERY_MV 3350
#define TWINR_FS_MIN_CLEARANCE_MM 350
#define TWINR_FS_MIN_UP_CLEARANCE_MM 250
#define TWINR_FS_DESCENT_RATE_MMPS 120
#define TWINR_FS_MAX_REPEL_VELOCITY_MMPS 150
#define TWINR_FS_BRAKE_HOLD_MS 200
#define TWINR_FS_RANGE_INVALID_MM 32000U
#define TWINR_FS_LANDING_FLOOR_MM 80U
#define TWINR_FS_TOUCHDOWN_HEIGHT_MM 30U
#define TWINR_FS_TOUCHDOWN_CONFIRM_MM 50U
#define TWINR_FS_TOUCHDOWN_CONFIRM_SAMPLES 3U
#define TWINR_FS_MIN_ACTIVE_ALTITUDE_M 0.08f
#define TWINR_FS_MAX_TARGET_Z_M 1.20f
#define TWINR_FS_COMMANDER_PRIORITY COMMANDER_PRIORITY_EXTRX

typedef enum {
  twinrFsStateDisabled = 0,
  twinrFsStateMonitoring = 1,
  twinrFsStateFailsafeBrake = 2,
  twinrFsStateFailsafeDescend = 3,
  twinrFsStateTouchdownConfirm = 4,
  twinrFsStateLanded = 5,
} twinrFsState_t;

typedef enum {
  twinrFsReasonNone = 0,
  twinrFsReasonHeartbeatLoss = 1,
  twinrFsReasonLowBattery = 2,
  twinrFsReasonCriticalBattery = 3,
  twinrFsReasonClearance = 4,
  twinrFsReasonUpClearance = 5,
  twinrFsReasonManualDisable = 6,
} twinrFsReason_t;

typedef struct {
  uint8_t version;
  uint8_t packetKind;
  uint8_t flags;
  uint8_t reserved;
  uint16_t sessionId;
  uint16_t heartbeatTimeoutMs;
  uint16_t lowBatteryMv;
  uint16_t criticalBatteryMv;
  uint16_t minClearanceMm;
  uint16_t minUpClearanceMm;
  uint16_t descentRateMmps;
  uint16_t maxRepelVelocityMmps;
  uint16_t brakeHoldMs;
} __attribute__((packed)) twinrFsHeartbeatPacket_t;

typedef struct {
  uint8_t version;
  uint8_t packetKind;
  uint8_t state;
  uint8_t reason;
  uint16_t sessionId;
  uint16_t heartbeatAgeMs;
  uint16_t vbatMv;
  uint16_t minClearanceMm;
  uint16_t downRangeMm;
} __attribute__((packed)) twinrFsStatusPacket_t;

static uint8_t twinrFsProtocolVersion = TWINR_FS_PROTOCOL_VERSION;
static uint8_t twinrFsEnable = 0;
static uint8_t twinrFsRequireClearance = 1;
static uint16_t twinrFsSessionId = 0;
static uint16_t twinrFsHeartbeatTimeoutMs = TWINR_FS_HEARTBEAT_TIMEOUT_MS;
static uint16_t twinrFsLowBatteryMv = TWINR_FS_LOW_BATTERY_MV;
static uint16_t twinrFsCriticalBatteryMv = TWINR_FS_CRITICAL_BATTERY_MV;
static uint16_t twinrFsMinClearanceMm = TWINR_FS_MIN_CLEARANCE_MM;
static uint16_t twinrFsMinUpClearanceMm = TWINR_FS_MIN_UP_CLEARANCE_MM;
static uint16_t twinrFsDescentRateMmps = TWINR_FS_DESCENT_RATE_MMPS;
static uint16_t twinrFsMaxRepelVelocityMmps = TWINR_FS_MAX_REPEL_VELOCITY_MMPS;
static uint16_t twinrFsBrakeHoldMs = TWINR_FS_BRAKE_HOLD_MS;
static uint8_t twinrFsState = twinrFsStateDisabled;
static uint8_t twinrFsReason = twinrFsReasonNone;
static uint16_t twinrFsHeartbeatAgeMs = 0;
static uint16_t twinrFsLastVbatMv = 0;
static uint16_t twinrFsLastMinClearanceMm = 0;
static uint16_t twinrFsLastDownRangeMm = 0;
static uint8_t twinrFsControlActive = 0;

static TickType_t twinrFsLastHeartbeatTick = 0;
static TickType_t twinrFsLastStateTick = 0;
static TickType_t twinrFsLastStatusTick = 0;
static float twinrFsFailsafeTargetZ = 0.0f;
static uint8_t twinrFsTouchdownConfirmCount = 0;

static logVarId_t twinrFsFrontId;
static logVarId_t twinrFsBackId;
static logVarId_t twinrFsLeftId;
static logVarId_t twinrFsRightId;
static logVarId_t twinrFsUpId;
static logVarId_t twinrFsDownId;
static logVarId_t twinrFsVbatMvId;
static logVarId_t twinrFsStateEstimateZId;

static uint16_t clampUint16(const uint32_t value)
{
  return value > UINT16_MAX ? UINT16_MAX : (uint16_t)value;
}

static float clampFloat(const float value, const float lower, const float upper)
{
  if (value < lower) {
    return lower;
  }
  if (value > upper) {
    return upper;
  }
  return value;
}

static bool rangeIsValid(const uint16_t rangeMm)
{
  return rangeMm > 0U && rangeMm < TWINR_FS_RANGE_INVALID_MM;
}

static uint16_t readRangeMm(const logVarId_t id)
{
  if (!logVarIdIsValid(id)) {
    return 0U;
  }
  return clampUint16(logGetUint(id));
}

static uint16_t readVbatMv(void)
{
  if (!logVarIdIsValid(twinrFsVbatMvId)) {
    return 0U;
  }
  return clampUint16(logGetUint(twinrFsVbatMvId));
}

static float readStateEstimateZ(void)
{
  if (!logVarIdIsValid(twinrFsStateEstimateZId)) {
    return 0.0f;
  }
  const float z = logGetFloat(twinrFsStateEstimateZId);
  if (!isfinite(z) || z < 0.0f) {
    return 0.0f;
  }
  return z;
}

static void setHoverSetpoint(setpoint_t *setpoint, const float vx, const float vy, const float z)
{
  memset(setpoint, 0, sizeof(setpoint_t));
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;
  setpoint->mode.yaw = modeVelocity;
  setpoint->attitudeRate.yaw = 0.0f;
  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;
  setpoint->velocity_body = true;
}

static void sendFailsafeSetpoint(const float vx, const float vy, const float z)
{
  static setpoint_t setpoint;
  setHoverSetpoint(&setpoint, vx, vy, z);
  commanderSetSetpoint(&setpoint, TWINR_FS_COMMANDER_PRIORITY);
}

static float computeRepelComponent(const uint16_t rangeMm, const uint16_t thresholdMm, const float direction, const float maxVelocity)
{
  if (!rangeIsValid(rangeMm) || rangeMm >= thresholdMm || thresholdMm == 0U) {
    return 0.0f;
  }

  const float ratio = ((float)(thresholdMm - rangeMm)) / (float)thresholdMm;
  return direction * ratio * maxVelocity;
}

static void computeRepelVelocity(
  const uint16_t frontMm,
  const uint16_t backMm,
  const uint16_t leftMm,
  const uint16_t rightMm,
  float *vx,
  float *vy)
{
  const float maxVelocity = ((float)twinrFsMaxRepelVelocityMmps) / 1000.0f;
  *vx = computeRepelComponent(frontMm, twinrFsMinClearanceMm, -1.0f, maxVelocity)
      + computeRepelComponent(backMm, twinrFsMinClearanceMm, 1.0f, maxVelocity);
  *vy = computeRepelComponent(leftMm, twinrFsMinClearanceMm, -1.0f, maxVelocity)
      + computeRepelComponent(rightMm, twinrFsMinClearanceMm, 1.0f, maxVelocity);
}

static uint16_t minClearanceMm(
  const uint16_t frontMm,
  const uint16_t backMm,
  const uint16_t leftMm,
  const uint16_t rightMm,
  const uint16_t upMm)
{
  uint16_t result = 0U;
  const uint16_t samples[] = {frontMm, backMm, leftMm, rightMm, upMm};
  for (size_t index = 0; index < sizeof(samples) / sizeof(samples[0]); ++index) {
    const uint16_t sample = samples[index];
    if (!rangeIsValid(sample)) {
      continue;
    }
    if (result == 0U || sample < result) {
      result = sample;
    }
  }
  return result;
}

static bool clearanceViolated(
  const uint16_t frontMm,
  const uint16_t backMm,
  const uint16_t leftMm,
  const uint16_t rightMm,
  const uint16_t upMm,
  twinrFsReason_t *reason)
{
  if (!twinrFsRequireClearance) {
    return false;
  }
  if (rangeIsValid(upMm) && upMm < twinrFsMinUpClearanceMm) {
    *reason = twinrFsReasonUpClearance;
    return true;
  }
  const uint16_t lateralRanges[] = {frontMm, backMm, leftMm, rightMm};
  for (size_t index = 0; index < sizeof(lateralRanges) / sizeof(lateralRanges[0]); ++index) {
    if (rangeIsValid(lateralRanges[index]) && lateralRanges[index] < twinrFsMinClearanceMm) {
      *reason = twinrFsReasonClearance;
      return true;
    }
  }
  return false;
}

static bool heartbeatExpired(const TickType_t now)
{
  if (!twinrFsEnable || twinrFsLastHeartbeatTick == 0 || twinrFsHeartbeatTimeoutMs == 0U) {
    return false;
  }
  return (now - twinrFsLastHeartbeatTick) > M2T(twinrFsHeartbeatTimeoutMs);
}

static void maybeSendStatus(const TickType_t now, const bool force)
{
  if (!force && (now - twinrFsLastStatusTick) < M2T(TWINR_FS_STATUS_PERIOD_MS)) {
    return;
  }

  twinrFsStatusPacket_t packet = {
    .version = TWINR_FS_PROTOCOL_VERSION,
    .packetKind = TWINR_FS_PACKET_KIND_STATUS,
    .state = twinrFsState,
    .reason = twinrFsReason,
    .sessionId = twinrFsSessionId,
    .heartbeatAgeMs = twinrFsHeartbeatAgeMs,
    .vbatMv = twinrFsLastVbatMv,
    .minClearanceMm = twinrFsLastMinClearanceMm,
    .downRangeMm = twinrFsLastDownRangeMm,
  };

  appchannelSendDataPacket(&packet, sizeof(packet));
  twinrFsLastStatusTick = now;
}

static void setState(const twinrFsState_t state, const twinrFsReason_t reason, const TickType_t now, const bool forceStatus)
{
  const bool changed = (twinrFsState != (uint8_t)state) || (twinrFsReason != (uint8_t)reason);
  twinrFsState = (uint8_t)state;
  twinrFsReason = (uint8_t)reason;
  if (changed) {
    twinrFsLastStateTick = now;
  }
  maybeSendStatus(now, forceStatus || changed);
}

static void triggerFailsafe(const twinrFsReason_t reason, const TickType_t now, const float stateEstimateZ, const uint16_t downMm)
{
  if (twinrFsControlActive) {
    return;
  }

  float targetZ = stateEstimateZ;
  if (targetZ < TWINR_FS_MIN_ACTIVE_ALTITUDE_M && rangeIsValid(downMm)) {
    targetZ = ((float)downMm) / 1000.0f;
  }
  if (targetZ < TWINR_FS_MIN_ACTIVE_ALTITUDE_M) {
    targetZ = TWINR_FS_MIN_ACTIVE_ALTITUDE_M;
  }

  twinrFsFailsafeTargetZ = clampFloat(targetZ, TWINR_FS_MIN_ACTIVE_ALTITUDE_M, TWINR_FS_MAX_TARGET_Z_M);
  twinrFsTouchdownConfirmCount = 0U;
  twinrFsControlActive = 1U;
  setState(twinrFsStateFailsafeBrake, reason, now, true);
  DEBUG_PRINT("Failsafe triggered reason=%u targetZ=%.3f\n", (unsigned)reason, (double)twinrFsFailsafeTargetZ);
}

static void applyHeartbeatPacket(const twinrFsHeartbeatPacket_t *packet, const TickType_t now)
{
  if (packet->version != TWINR_FS_PROTOCOL_VERSION || packet->packetKind != TWINR_FS_PACKET_KIND_HEARTBEAT) {
    return;
  }

  twinrFsEnable = (packet->flags & TWINR_FS_FLAG_ENABLE) ? 1U : 0U;
  twinrFsRequireClearance = (packet->flags & TWINR_FS_FLAG_REQUIRE_CLEARANCE) ? 1U : 0U;
  twinrFsSessionId = packet->sessionId;

  if (packet->heartbeatTimeoutMs > 0U) {
    twinrFsHeartbeatTimeoutMs = packet->heartbeatTimeoutMs;
  }
  if (packet->lowBatteryMv > 0U) {
    twinrFsLowBatteryMv = packet->lowBatteryMv;
  }
  if (packet->criticalBatteryMv > 0U) {
    twinrFsCriticalBatteryMv = packet->criticalBatteryMv;
  }
  if (packet->minClearanceMm > 0U) {
    twinrFsMinClearanceMm = packet->minClearanceMm;
  }
  if (packet->minUpClearanceMm > 0U) {
    twinrFsMinUpClearanceMm = packet->minUpClearanceMm;
  }
  if (packet->descentRateMmps > 0U) {
    twinrFsDescentRateMmps = packet->descentRateMmps;
  }
  if (packet->maxRepelVelocityMmps > 0U) {
    twinrFsMaxRepelVelocityMmps = packet->maxRepelVelocityMmps;
  }
  if (packet->brakeHoldMs > 0U) {
    twinrFsBrakeHoldMs = packet->brakeHoldMs;
  }

  twinrFsLastHeartbeatTick = now;
  twinrFsHeartbeatAgeMs = 0U;

  if (!twinrFsEnable && !twinrFsControlActive) {
    setState(twinrFsStateDisabled, twinrFsReasonManualDisable, now, true);
  } else if (!twinrFsControlActive) {
    setState(twinrFsStateMonitoring, twinrFsReasonNone, now, true);
  }
}

static void processHeartbeatPackets(const TickType_t now)
{
  twinrFsHeartbeatPacket_t packet;
  while (appchannelReceiveDataPacket(&packet, sizeof(packet), 0) == sizeof(packet)) {
    applyHeartbeatPacket(&packet, now);
  }
}

static void runFailsafeControl(
  const TickType_t now,
  const float stateEstimateZ,
  const uint16_t frontMm,
  const uint16_t backMm,
  const uint16_t leftMm,
  const uint16_t rightMm,
  const uint16_t downMm,
  const bool isFlying)
{
  float vx = 0.0f;
  float vy = 0.0f;
  computeRepelVelocity(frontMm, backMm, leftMm, rightMm, &vx, &vy);

  if (twinrFsState == twinrFsStateFailsafeBrake) {
    sendFailsafeSetpoint(vx, vy, twinrFsFailsafeTargetZ);
    if ((now - twinrFsLastStateTick) >= M2T(twinrFsBrakeHoldMs)) {
      setState(twinrFsStateFailsafeDescend, twinrFsReason, now, true);
    }
    return;
  }

  if (twinrFsState == twinrFsStateFailsafeDescend) {
    const float descentPerStep = (((float)twinrFsDescentRateMmps) / 1000.0f) * (((float)TWINR_FS_LOOP_PERIOD_MS) / 1000.0f);
    if (twinrFsFailsafeTargetZ > 0.0f) {
      twinrFsFailsafeTargetZ = fmaxf(0.0f, twinrFsFailsafeTargetZ - descentPerStep);
    }
    if (rangeIsValid(downMm) && downMm <= TWINR_FS_LANDING_FLOOR_MM) {
      twinrFsFailsafeTargetZ = fminf(twinrFsFailsafeTargetZ, ((float)TWINR_FS_LANDING_FLOOR_MM) / 1000.0f);
    }
    sendFailsafeSetpoint(vx, vy, twinrFsFailsafeTargetZ);
    if ((rangeIsValid(downMm) && downMm <= TWINR_FS_TOUCHDOWN_HEIGHT_MM) || stateEstimateZ <= 0.03f) {
      setState(twinrFsStateTouchdownConfirm, twinrFsReason, now, true);
    }
    return;
  }

  if (twinrFsState == twinrFsStateTouchdownConfirm) {
    const bool touchdownByRange = rangeIsValid(downMm) && downMm <= TWINR_FS_TOUCHDOWN_CONFIRM_MM;
    const bool touchdownBySupervisor = !isFlying;
    sendFailsafeSetpoint(0.0f, 0.0f, 0.0f);
    if (touchdownByRange || touchdownBySupervisor) {
      if (twinrFsTouchdownConfirmCount < UINT8_MAX) {
        twinrFsTouchdownConfirmCount += 1U;
      }
    } else {
      twinrFsTouchdownConfirmCount = 0U;
    }
    if (twinrFsTouchdownConfirmCount >= TWINR_FS_TOUCHDOWN_CONFIRM_SAMPLES) {
      if (supervisorRequestArming(false)) {
        commanderRelaxPriority();
        twinrFsControlActive = 0U;
        setState(twinrFsStateLanded, twinrFsReason, now, true);
      }
    }
  }
}

void appMain(void)
{
  vTaskDelay(M2T(1000));

  twinrFsFrontId = logGetVarId("range", "front");
  twinrFsBackId = logGetVarId("range", "back");
  twinrFsLeftId = logGetVarId("range", "left");
  twinrFsRightId = logGetVarId("range", "right");
  twinrFsUpId = logGetVarId("range", "up");
  twinrFsDownId = logGetVarId("range", "zrange");
  twinrFsVbatMvId = logGetVarId("pm", "vbatMV");
  twinrFsStateEstimateZId = logGetVarId("stateEstimate", "z");

  TickType_t lastWakeTime = xTaskGetTickCount();
  DEBUG_PRINT("Twinr on-device failsafe app ready\n");

  while (1) {
    vTaskDelayUntil(&lastWakeTime, M2T(TWINR_FS_LOOP_PERIOD_MS));
    const TickType_t now = xTaskGetTickCount();

    processHeartbeatPackets(now);

    const uint16_t frontMm = readRangeMm(twinrFsFrontId);
    const uint16_t backMm = readRangeMm(twinrFsBackId);
    const uint16_t leftMm = readRangeMm(twinrFsLeftId);
    const uint16_t rightMm = readRangeMm(twinrFsRightId);
    const uint16_t upMm = readRangeMm(twinrFsUpId);
    const uint16_t downMm = readRangeMm(twinrFsDownId);
    const uint16_t vbatMv = readVbatMv();
    const float stateEstimateZ = readStateEstimateZ();
    const bool isFlying = supervisorIsFlying();
    const bool airborne = isFlying || (rangeIsValid(downMm) && downMm > TWINR_FS_LANDING_FLOOR_MM) || stateEstimateZ > TWINR_FS_MIN_ACTIVE_ALTITUDE_M;

    twinrFsLastVbatMv = vbatMv;
    twinrFsLastMinClearanceMm = minClearanceMm(frontMm, backMm, leftMm, rightMm, upMm);
    twinrFsLastDownRangeMm = downMm;
    twinrFsHeartbeatAgeMs = twinrFsLastHeartbeatTick == 0 ? 0U : clampUint16(T2M(now - twinrFsLastHeartbeatTick));

    if (twinrFsControlActive) {
      runFailsafeControl(now, stateEstimateZ, frontMm, backMm, leftMm, rightMm, downMm, isFlying);
      maybeSendStatus(now, false);
      continue;
    }

    if (!twinrFsEnable) {
      setState(twinrFsStateDisabled, twinrFsReasonManualDisable, now, false);
      continue;
    }

    if (!airborne) {
      setState(twinrFsStateMonitoring, twinrFsReasonNone, now, false);
      maybeSendStatus(now, false);
      continue;
    }

    twinrFsReason_t triggerReason = twinrFsReasonNone;
    if (vbatMv > 0U && vbatMv <= twinrFsCriticalBatteryMv) {
      triggerReason = twinrFsReasonCriticalBattery;
    } else if (clearanceViolated(frontMm, backMm, leftMm, rightMm, upMm, &triggerReason)) {
      /* triggerReason already set */
    } else if (heartbeatExpired(now)) {
      triggerReason = twinrFsReasonHeartbeatLoss;
    } else if (vbatMv > 0U && vbatMv <= twinrFsLowBatteryMv) {
      triggerReason = twinrFsReasonLowBattery;
    }

    if (triggerReason != twinrFsReasonNone) {
      triggerFailsafe(triggerReason, now, stateEstimateZ, downMm);
      runFailsafeControl(now, stateEstimateZ, frontMm, backMm, leftMm, rightMm, downMm, isFlying);
      continue;
    }

    setState(twinrFsStateMonitoring, twinrFsReasonNone, now, false);
    maybeSendStatus(now, false);
  }
}

LOG_GROUP_START(twinrFs)
LOG_ADD_CORE(LOG_UINT8, state, &twinrFsState)
LOG_ADD_CORE(LOG_UINT8, reason, &twinrFsReason)
LOG_ADD(LOG_UINT16, heartbeatAgeMs, &twinrFsHeartbeatAgeMs)
LOG_ADD(LOG_UINT16, vbatMv, &twinrFsLastVbatMv)
LOG_ADD(LOG_UINT16, minClearanceMm, &twinrFsLastMinClearanceMm)
LOG_ADD(LOG_UINT16, downRangeMm, &twinrFsLastDownRangeMm)
LOG_GROUP_STOP(twinrFs)

PARAM_GROUP_START(twinrFs)
PARAM_ADD_CORE(PARAM_UINT8 | PARAM_RONLY, protocolVersion, &twinrFsProtocolVersion)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, enable, &twinrFsEnable)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, requireClearance, &twinrFsRequireClearance)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, sessionId, &twinrFsSessionId)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, heartbeatTimeoutMs, &twinrFsHeartbeatTimeoutMs)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, lowBatteryMv, &twinrFsLowBatteryMv)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, criticalBatteryMv, &twinrFsCriticalBatteryMv)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, minClearanceMm, &twinrFsMinClearanceMm)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, minUpClearanceMm, &twinrFsMinUpClearanceMm)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, descentRateMmps, &twinrFsDescentRateMmps)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, maxRepelVelocityMmps, &twinrFsMaxRepelVelocityMmps)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, brakeHoldMs, &twinrFsBrakeHoldMs)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, state, &twinrFsState)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, reason, &twinrFsReason)
PARAM_GROUP_STOP(twinrFs)
