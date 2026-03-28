// # CHANGELOG: 2026-03-27
// # BUG-1: Appchannel RX parsing accepted silently cropped packets and stopped draining the RX queue on the first non-heartbeat packet, which can starve valid heartbeats and cause false failsafe triggers.
// # BUG-2: Repel velocities from multiple directions could sum to a planar speed above maxRepelVelocityMmps, causing stronger-than-configured corner pushback.
// # BUG-3: Battery and clearance triggers reacted to single transient samples, causing false emergency landings under voltage sag or ToF spikes.
// # SEC-1: Heartbeats from a different session could reconfigure or disable the failsafe in-flight. The implementation now binds to a session, rejects cross-session packets while airborne, and freezes config changes during flight.
// # SEC-2: In-flight remote disable is no longer accepted. # BREAKING: disable/reconfigure requests are only applied while grounded.
// # IMP-1: Added protocol v2 heartbeat support with monotonic sequence numbers and configurable debounce ticks, while keeping protocol v1 compatibility.
// # IMP-2: Added event-trigger observability and RX overflow/reject counters for post-flight forensics and safer field debugging.

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
#include "eventtrigger.h"
#include "FreeRTOS.h"
#include "log.h"
#include "param.h"
#include "supervisor.h"
#include "task.h"

#define DEBUG_MODULE "TWINRFS"

#define TWINR_FS_PROTOCOL_VERSION 2U
#define TWINR_FS_PROTOCOL_VERSION_LEGACY 1U
#define TWINR_FS_PACKET_KIND_HEARTBEAT 1U
#define TWINR_FS_PACKET_KIND_STATUS 2U
#define TWINR_FS_FLAG_ENABLE (1U << 0)
#define TWINR_FS_FLAG_REQUIRE_CLEARANCE (1U << 1)

#define TWINR_FS_LOOP_PERIOD_MS 20U
#define TWINR_FS_STATUS_PERIOD_MS 100U
#define TWINR_FS_HEARTBEAT_TIMEOUT_MS 350U
#define TWINR_FS_HEARTBEAT_TIMEOUT_MIN_MS 100U
#define TWINR_FS_HEARTBEAT_TIMEOUT_MAX_MS 5000U
#define TWINR_FS_LOW_BATTERY_MV 3550U
#define TWINR_FS_CRITICAL_BATTERY_MV 3350U
#define TWINR_FS_BATTERY_THRESHOLD_MIN_MV 3000U
#define TWINR_FS_BATTERY_THRESHOLD_MAX_MV 4300U
#define TWINR_FS_BATTERY_HYSTERESIS_MV 60U
#define TWINR_FS_MIN_CLEARANCE_MM 350U
#define TWINR_FS_MIN_UP_CLEARANCE_MM 250U
#define TWINR_FS_CLEARANCE_MIN_MM 100U
#define TWINR_FS_CLEARANCE_MAX_MM 3000U
#define TWINR_FS_DESCENT_RATE_MMPS 120U
#define TWINR_FS_DESCENT_RATE_MIN_MMPS 40U
#define TWINR_FS_DESCENT_RATE_MAX_MMPS 500U
#define TWINR_FS_MAX_REPEL_VELOCITY_MMPS 150U
#define TWINR_FS_MAX_REPEL_VELOCITY_LIMIT_MMPS 500U
#define TWINR_FS_BRAKE_HOLD_MS 200U
#define TWINR_FS_BRAKE_HOLD_MAX_MS 2000U
#define TWINR_FS_RANGE_INVALID_MM 32000U
#define TWINR_FS_LANDING_FLOOR_MM 80U
#define TWINR_FS_TOUCHDOWN_HEIGHT_MM 30U
#define TWINR_FS_TOUCHDOWN_CONFIRM_MM 50U
#define TWINR_FS_TOUCHDOWN_CONFIRM_SAMPLES 3U
#define TWINR_FS_MIN_ACTIVE_ALTITUDE_M 0.08f
#define TWINR_FS_MAX_TARGET_Z_M 1.20f
#define TWINR_FS_COMMANDER_PRIORITY COMMANDER_PRIORITY_EXTRX
#define TWINR_FS_LOW_BATTERY_DEBOUNCE_TICKS_DEFAULT 25U
#define TWINR_FS_CRITICAL_BATTERY_DEBOUNCE_TICKS_DEFAULT 6U
#define TWINR_FS_CLEARANCE_DEBOUNCE_TICKS_DEFAULT 2U
#define TWINR_FS_DEBOUNCE_TICKS_MIN 1U
#define TWINR_FS_DEBOUNCE_TICKS_MAX 50U

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

typedef enum {
  twinrFsRejectNone = 0,
  twinrFsRejectMalformed = 1,
  twinrFsRejectWrongKind = 2,
  twinrFsRejectUnsupportedVersion = 3,
  twinrFsRejectSessionMismatch = 4,
  twinrFsRejectStaleSequence = 5,
  twinrFsRejectInFlightReconfigure = 6,
  twinrFsRejectInFlightDisable = 7,
  twinrFsRejectDowngrade = 8,
} twinrFsReject_t;

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
} __attribute__((packed)) twinrFsHeartbeatPacketV1_t;

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
  uint8_t lowBatteryDebounceTicks;
  uint8_t criticalBatteryDebounceTicks;
  uint8_t clearanceDebounceTicks;
  uint8_t reserved2;
  uint32_t sequence;
} __attribute__((packed)) twinrFsHeartbeatPacketV2_t;

typedef struct {
  uint8_t version;
  uint8_t flags;
  uint16_t sessionId;
  uint16_t heartbeatTimeoutMs;
  uint16_t lowBatteryMv;
  uint16_t criticalBatteryMv;
  uint16_t minClearanceMm;
  uint16_t minUpClearanceMm;
  uint16_t descentRateMmps;
  uint16_t maxRepelVelocityMmps;
  uint16_t brakeHoldMs;
  uint8_t lowBatteryDebounceTicks;
  uint8_t criticalBatteryDebounceTicks;
  uint8_t clearanceDebounceTicks;
  bool hasSequence;
  uint32_t sequence;
} twinrFsHeartbeatConfig_t;

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

EVENTTRIGGER(twinrFsTrigger,
             uint8, reason,
             uint16, sessionId,
             uint16, vbatMv,
             uint16, minClearanceMm,
             uint16, downRangeMm)

EVENTTRIGGER(twinrFsReject,
             uint8, rejectCode,
             uint8, version,
             uint8, packetKind,
             uint16, sessionId,
             uint16, detail)

EVENTTRIGGER(twinrFsOverflow,
             uint16, sessionId,
             uint16, heartbeatAgeMs)

EVENTTRIGGER(twinrFsLanded,
             uint8, reason,
             uint16, sessionId,
             uint16, downRangeMm)

static uint8_t twinrFsProtocolVersion = TWINR_FS_PROTOCOL_VERSION;
static uint8_t twinrFsEnable = 0;
static uint8_t twinrFsRequireClearance = 1;
static uint16_t twinrFsSessionId = 0;
static uint8_t twinrFsSessionBound = 0;
static uint8_t twinrFsPacketVersionInUse = TWINR_FS_PROTOCOL_VERSION_LEGACY;
static uint16_t twinrFsHeartbeatTimeoutMs = TWINR_FS_HEARTBEAT_TIMEOUT_MS;
static uint16_t twinrFsLowBatteryMv = TWINR_FS_LOW_BATTERY_MV;
static uint16_t twinrFsCriticalBatteryMv = TWINR_FS_CRITICAL_BATTERY_MV;
static uint16_t twinrFsMinClearanceMm = TWINR_FS_MIN_CLEARANCE_MM;
static uint16_t twinrFsMinUpClearanceMm = TWINR_FS_MIN_UP_CLEARANCE_MM;
static uint16_t twinrFsDescentRateMmps = TWINR_FS_DESCENT_RATE_MMPS;
static uint16_t twinrFsMaxRepelVelocityMmps = TWINR_FS_MAX_REPEL_VELOCITY_MMPS;
static uint16_t twinrFsBrakeHoldMs = TWINR_FS_BRAKE_HOLD_MS;
static uint8_t twinrFsLowBatteryDebounceTicks = TWINR_FS_LOW_BATTERY_DEBOUNCE_TICKS_DEFAULT;
static uint8_t twinrFsCriticalBatteryDebounceTicks = TWINR_FS_CRITICAL_BATTERY_DEBOUNCE_TICKS_DEFAULT;
static uint8_t twinrFsClearanceDebounceTicks = TWINR_FS_CLEARANCE_DEBOUNCE_TICKS_DEFAULT;
static uint8_t twinrFsState = twinrFsStateDisabled;
static uint8_t twinrFsReason = twinrFsReasonNone;
static uint16_t twinrFsHeartbeatAgeMs = 0;
static uint16_t twinrFsLastVbatMv = 0;
static uint16_t twinrFsLastMinClearanceMm = 0;
static uint16_t twinrFsLastDownRangeMm = 0;
static uint8_t twinrFsLastPmState = 0;
static uint8_t twinrFsControlActive = 0;
static uint8_t twinrFsLastRejectCode = twinrFsRejectNone;
static uint16_t twinrFsRejectedPacketCount = 0;
static uint16_t twinrFsOverflowCount = 0;
static uint32_t twinrFsLastHeartbeatSeq = 0;
static uint8_t twinrFsHeartbeatSeqValid = 0;

static TickType_t twinrFsLastHeartbeatTick = 0;
static TickType_t twinrFsLastStateTick = 0;
static TickType_t twinrFsLastStatusTick = 0;
static float twinrFsFailsafeTargetZ = 0.0f;
static uint8_t twinrFsTouchdownConfirmCount = 0;
static uint8_t twinrFsLowBatteryCount = 0;
static uint8_t twinrFsCriticalBatteryCount = 0;
static uint8_t twinrFsClearanceCount = 0;
static uint8_t twinrFsPendingClearanceReason = twinrFsReasonNone;

static logVarId_t twinrFsFrontId;
static logVarId_t twinrFsBackId;
static logVarId_t twinrFsLeftId;
static logVarId_t twinrFsRightId;
static logVarId_t twinrFsUpId;
static logVarId_t twinrFsDownId;
static logVarId_t twinrFsVbatMvId;
static logVarId_t twinrFsPmStateId;
static logVarId_t twinrFsStateEstimateZId;

static uint16_t clampUint16(const uint32_t value)
{
  return value > UINT16_MAX ? UINT16_MAX : (uint16_t)value;
}

static uint16_t clampUint16Range(const uint16_t value, const uint16_t lower, const uint16_t upper)
{
  if (value < lower) {
    return lower;
  }
  if (value > upper) {
    return upper;
  }
  return value;
}

static uint8_t clampUint8Range(const uint8_t value, const uint8_t lower, const uint8_t upper)
{
  if (value < lower) {
    return lower;
  }
  if (value > upper) {
    return upper;
  }
  return value;
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

static uint8_t readPmState(void)
{
  if (!logVarIdIsValid(twinrFsPmStateId)) {
    return 0U;
  }

  const int value = logGetInt(twinrFsPmStateId);
  if (value <= 0) {
    return 0U;
  }
  if (value >= UINT8_MAX) {
    return UINT8_MAX;
  }
  return (uint8_t)value;
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

static void resetTriggerCounters(void)
{
  twinrFsLowBatteryCount = 0U;
  twinrFsCriticalBatteryCount = 0U;
  twinrFsClearanceCount = 0U;
  twinrFsPendingClearanceReason = twinrFsReasonNone;
}

static void sanitizeHeartbeatConfig(twinrFsHeartbeatConfig_t *config)
{
  config->heartbeatTimeoutMs = clampUint16Range(config->heartbeatTimeoutMs,
                                                TWINR_FS_HEARTBEAT_TIMEOUT_MIN_MS,
                                                TWINR_FS_HEARTBEAT_TIMEOUT_MAX_MS);
  config->lowBatteryMv = clampUint16Range(config->lowBatteryMv,
                                          TWINR_FS_BATTERY_THRESHOLD_MIN_MV,
                                          TWINR_FS_BATTERY_THRESHOLD_MAX_MV);
  config->criticalBatteryMv = clampUint16Range(config->criticalBatteryMv,
                                               TWINR_FS_BATTERY_THRESHOLD_MIN_MV,
                                               TWINR_FS_BATTERY_THRESHOLD_MAX_MV);
  if (config->criticalBatteryMv >= config->lowBatteryMv) {
    config->criticalBatteryMv = (config->lowBatteryMv > 100U)
                                  ? (uint16_t)(config->lowBatteryMv - 100U)
                                  : config->lowBatteryMv;
  }

  config->minClearanceMm = clampUint16Range(config->minClearanceMm,
                                            TWINR_FS_CLEARANCE_MIN_MM,
                                            TWINR_FS_CLEARANCE_MAX_MM);
  config->minUpClearanceMm = clampUint16Range(config->minUpClearanceMm,
                                              TWINR_FS_CLEARANCE_MIN_MM,
                                              TWINR_FS_CLEARANCE_MAX_MM);
  config->descentRateMmps = clampUint16Range(config->descentRateMmps,
                                             TWINR_FS_DESCENT_RATE_MIN_MMPS,
                                             TWINR_FS_DESCENT_RATE_MAX_MMPS);
  config->maxRepelVelocityMmps = clampUint16Range(config->maxRepelVelocityMmps,
                                                  0U,
                                                  TWINR_FS_MAX_REPEL_VELOCITY_LIMIT_MMPS);
  config->brakeHoldMs = clampUint16Range(config->brakeHoldMs, 0U, TWINR_FS_BRAKE_HOLD_MAX_MS);
  config->lowBatteryDebounceTicks = clampUint8Range(config->lowBatteryDebounceTicks,
                                                    TWINR_FS_DEBOUNCE_TICKS_MIN,
                                                    TWINR_FS_DEBOUNCE_TICKS_MAX);
  config->criticalBatteryDebounceTicks = clampUint8Range(config->criticalBatteryDebounceTicks,
                                                         TWINR_FS_DEBOUNCE_TICKS_MIN,
                                                         TWINR_FS_DEBOUNCE_TICKS_MAX);
  config->clearanceDebounceTicks = clampUint8Range(config->clearanceDebounceTicks,
                                                   TWINR_FS_DEBOUNCE_TICKS_MIN,
                                                   TWINR_FS_DEBOUNCE_TICKS_MAX);
}

static twinrFsHeartbeatConfig_t currentHeartbeatConfig(void)
{
  twinrFsHeartbeatConfig_t config = {
    .version = twinrFsPacketVersionInUse,
    .flags = (uint8_t)((twinrFsEnable ? TWINR_FS_FLAG_ENABLE : 0U)
                     | (twinrFsRequireClearance ? TWINR_FS_FLAG_REQUIRE_CLEARANCE : 0U)),
    .sessionId = twinrFsSessionId,
    .heartbeatTimeoutMs = twinrFsHeartbeatTimeoutMs,
    .lowBatteryMv = twinrFsLowBatteryMv,
    .criticalBatteryMv = twinrFsCriticalBatteryMv,
    .minClearanceMm = twinrFsMinClearanceMm,
    .minUpClearanceMm = twinrFsMinUpClearanceMm,
    .descentRateMmps = twinrFsDescentRateMmps,
    .maxRepelVelocityMmps = twinrFsMaxRepelVelocityMmps,
    .brakeHoldMs = twinrFsBrakeHoldMs,
    .lowBatteryDebounceTicks = twinrFsLowBatteryDebounceTicks,
    .criticalBatteryDebounceTicks = twinrFsCriticalBatteryDebounceTicks,
    .clearanceDebounceTicks = twinrFsClearanceDebounceTicks,
    .hasSequence = false,
    .sequence = 0U,
  };
  return config;
}

static bool heartbeatConfigEqualsCurrent(const twinrFsHeartbeatConfig_t *config)
{
  const uint8_t currentFlags = (uint8_t)((twinrFsEnable ? TWINR_FS_FLAG_ENABLE : 0U)
                                       | (twinrFsRequireClearance ? TWINR_FS_FLAG_REQUIRE_CLEARANCE : 0U));

  return config->flags == currentFlags
      && config->heartbeatTimeoutMs == twinrFsHeartbeatTimeoutMs
      && config->lowBatteryMv == twinrFsLowBatteryMv
      && config->criticalBatteryMv == twinrFsCriticalBatteryMv
      && config->minClearanceMm == twinrFsMinClearanceMm
      && config->minUpClearanceMm == twinrFsMinUpClearanceMm
      && config->descentRateMmps == twinrFsDescentRateMmps
      && config->maxRepelVelocityMmps == twinrFsMaxRepelVelocityMmps
      && config->brakeHoldMs == twinrFsBrakeHoldMs
      && config->lowBatteryDebounceTicks == twinrFsLowBatteryDebounceTicks
      && config->criticalBatteryDebounceTicks == twinrFsCriticalBatteryDebounceTicks
      && config->clearanceDebounceTicks == twinrFsClearanceDebounceTicks;
}

static void noteReject(const twinrFsReject_t code,
                       const uint8_t version,
                       const uint8_t packetKind,
                       const uint16_t sessionId,
                       const uint16_t detail)
{
  twinrFsLastRejectCode = (uint8_t)code;
  if (twinrFsRejectedPacketCount < UINT16_MAX) {
    twinrFsRejectedPacketCount += 1U;
  }

  eventTrigger_twinrFsReject_payload.rejectCode = (uint8_t)code;
  eventTrigger_twinrFsReject_payload.version = version;
  eventTrigger_twinrFsReject_payload.packetKind = packetKind;
  eventTrigger_twinrFsReject_payload.sessionId = sessionId;
  eventTrigger_twinrFsReject_payload.detail = detail;
  eventTrigger(&eventTrigger_twinrFsReject);

  DEBUG_PRINT("Reject packet code=%u ver=%u kind=%u session=%u detail=%u\n",
              (unsigned)code,
              (unsigned)version,
              (unsigned)packetKind,
              (unsigned)sessionId,
              (unsigned)detail);
}

static bool parseHeartbeatPacket(const uint8_t *buffer,
                                 const size_t length,
                                 twinrFsHeartbeatConfig_t *config)
{
  if (length < 2U) {
    noteReject(twinrFsRejectMalformed, 0U, 0U, 0U, (uint16_t)length);
    return false;
  }

  const uint8_t version = buffer[0];
  const uint8_t packetKind = buffer[1];

  if (packetKind != TWINR_FS_PACKET_KIND_HEARTBEAT) {
    noteReject(twinrFsRejectWrongKind, version, packetKind, 0U, (uint16_t)length);
    return false;
  }

  *config = currentHeartbeatConfig();
  config->version = version;

  if (version == TWINR_FS_PROTOCOL_VERSION_LEGACY) {
    if (length != sizeof(twinrFsHeartbeatPacketV1_t)) {
      noteReject(twinrFsRejectMalformed, version, packetKind, 0U, (uint16_t)length);
      return false;
    }

    twinrFsHeartbeatPacketV1_t packet;
    memcpy(&packet, buffer, sizeof(packet));

    config->flags = packet.flags;
    config->sessionId = packet.sessionId;
    if (packet.heartbeatTimeoutMs > 0U) {
      config->heartbeatTimeoutMs = packet.heartbeatTimeoutMs;
    }
    if (packet.lowBatteryMv > 0U) {
      config->lowBatteryMv = packet.lowBatteryMv;
    }
    if (packet.criticalBatteryMv > 0U) {
      config->criticalBatteryMv = packet.criticalBatteryMv;
    }
    if (packet.minClearanceMm > 0U) {
      config->minClearanceMm = packet.minClearanceMm;
    }
    if (packet.minUpClearanceMm > 0U) {
      config->minUpClearanceMm = packet.minUpClearanceMm;
    }
    if (packet.descentRateMmps > 0U) {
      config->descentRateMmps = packet.descentRateMmps;
    }
    if (packet.maxRepelVelocityMmps > 0U) {
      config->maxRepelVelocityMmps = packet.maxRepelVelocityMmps;
    }
    if (packet.brakeHoldMs > 0U) {
      config->brakeHoldMs = packet.brakeHoldMs;
    }
    config->hasSequence = false;
  } else if (version == TWINR_FS_PROTOCOL_VERSION) {
    if (length != sizeof(twinrFsHeartbeatPacketV2_t)) {
      noteReject(twinrFsRejectMalformed, version, packetKind, 0U, (uint16_t)length);
      return false;
    }

    twinrFsHeartbeatPacketV2_t packet;
    memcpy(&packet, buffer, sizeof(packet));

    config->flags = packet.flags;
    config->sessionId = packet.sessionId;
    if (packet.heartbeatTimeoutMs > 0U) {
      config->heartbeatTimeoutMs = packet.heartbeatTimeoutMs;
    }
    if (packet.lowBatteryMv > 0U) {
      config->lowBatteryMv = packet.lowBatteryMv;
    }
    if (packet.criticalBatteryMv > 0U) {
      config->criticalBatteryMv = packet.criticalBatteryMv;
    }
    if (packet.minClearanceMm > 0U) {
      config->minClearanceMm = packet.minClearanceMm;
    }
    if (packet.minUpClearanceMm > 0U) {
      config->minUpClearanceMm = packet.minUpClearanceMm;
    }
    if (packet.descentRateMmps > 0U) {
      config->descentRateMmps = packet.descentRateMmps;
    }
    if (packet.maxRepelVelocityMmps > 0U) {
      config->maxRepelVelocityMmps = packet.maxRepelVelocityMmps;
    }
    if (packet.brakeHoldMs > 0U) {
      config->brakeHoldMs = packet.brakeHoldMs;
    }
    if (packet.lowBatteryDebounceTicks > 0U) {
      config->lowBatteryDebounceTicks = packet.lowBatteryDebounceTicks;
    }
    if (packet.criticalBatteryDebounceTicks > 0U) {
      config->criticalBatteryDebounceTicks = packet.criticalBatteryDebounceTicks;
    }
    if (packet.clearanceDebounceTicks > 0U) {
      config->clearanceDebounceTicks = packet.clearanceDebounceTicks;
    }
    config->hasSequence = true;
    config->sequence = packet.sequence;
  } else {
    noteReject(twinrFsRejectUnsupportedVersion, version, packetKind, 0U, (uint16_t)length);
    return false;
  }

  sanitizeHeartbeatConfig(config);
  return true;
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

static float computeRepelComponent(const uint16_t rangeMm,
                                   const uint16_t thresholdMm,
                                   const float direction,
                                   const float maxVelocity)
{
  if (!rangeIsValid(rangeMm) || rangeMm >= thresholdMm || thresholdMm == 0U) {
    return 0.0f;
  }

  const float ratio = ((float)(thresholdMm - rangeMm)) / (float)thresholdMm;
  return direction * ratio * maxVelocity;
}

static void clampPlanarVelocity(float *vx, float *vy, const float maxVelocity)
{
  const float norm = sqrtf((*vx * *vx) + (*vy * *vy));
  if (norm <= maxVelocity || norm <= 1e-6f) {
    return;
  }

  const float scale = maxVelocity / norm;
  *vx *= scale;
  *vy *= scale;
}

static void computeRepelVelocity(const uint16_t frontMm,
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
  clampPlanarVelocity(vx, vy, maxVelocity);
}

static uint16_t minClearanceMm(const uint16_t frontMm,
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

static bool clearanceViolated(const uint16_t frontMm,
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

static bool batteryCounterReached(const bool active,
                                  const uint16_t measurement,
                                  const uint16_t threshold,
                                  const uint16_t hysteresis,
                                  uint8_t *count,
                                  const uint8_t debounceTicks)
{
  if (active) {
    if (*count < UINT8_MAX) {
      *count += 1U;
    }
  } else if (measurement == 0U || measurement > (uint16_t)(threshold + hysteresis)) {
    *count = 0U;
  }

  return *count >= debounceTicks;
}

static bool batteryViolated(const uint16_t vbatMv,
                            const uint8_t pmState,
                            twinrFsReason_t *reason)
{
  const bool pmCritical = pmState >= 3U;
  const bool criticalActive = pmCritical || (vbatMv > 0U && vbatMv <= twinrFsCriticalBatteryMv);
  const bool lowActive = vbatMv > 0U && vbatMv <= twinrFsLowBatteryMv;

  const bool criticalReached = batteryCounterReached(criticalActive,
                                                     vbatMv,
                                                     twinrFsCriticalBatteryMv,
                                                     TWINR_FS_BATTERY_HYSTERESIS_MV,
                                                     &twinrFsCriticalBatteryCount,
                                                     twinrFsCriticalBatteryDebounceTicks);
  const bool lowReached = batteryCounterReached(lowActive,
                                                vbatMv,
                                                twinrFsLowBatteryMv,
                                                TWINR_FS_BATTERY_HYSTERESIS_MV,
                                                &twinrFsLowBatteryCount,
                                                twinrFsLowBatteryDebounceTicks);

  if (criticalReached) {
    *reason = twinrFsReasonCriticalBattery;
    return true;
  }

  if (lowReached) {
    *reason = twinrFsReasonLowBattery;
    return true;
  }

  return false;
}

static bool clearanceDebounced(const uint16_t frontMm,
                               const uint16_t backMm,
                               const uint16_t leftMm,
                               const uint16_t rightMm,
                               const uint16_t upMm,
                               twinrFsReason_t *reason)
{
  twinrFsReason_t currentReason = twinrFsReasonNone;
  if (clearanceViolated(frontMm, backMm, leftMm, rightMm, upMm, &currentReason)) {
    if (twinrFsClearanceCount < UINT8_MAX) {
      twinrFsClearanceCount += 1U;
    }
    twinrFsPendingClearanceReason = (uint8_t)currentReason;
  } else {
    twinrFsClearanceCount = 0U;
    twinrFsPendingClearanceReason = twinrFsReasonNone;
  }

  if (twinrFsClearanceCount >= twinrFsClearanceDebounceTicks) {
    *reason = (twinrFsReason_t)twinrFsPendingClearanceReason;
    return true;
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
    .version = twinrFsPacketVersionInUse,
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

static void setState(const twinrFsState_t state,
                     const twinrFsReason_t reason,
                     const TickType_t now,
                     const bool forceStatus)
{
  const bool changed = (twinrFsState != (uint8_t)state) || (twinrFsReason != (uint8_t)reason);
  twinrFsState = (uint8_t)state;
  twinrFsReason = (uint8_t)reason;
  if (changed) {
    twinrFsLastStateTick = now;
  }
  maybeSendStatus(now, forceStatus || changed);
}

static void recordTriggerEvent(const twinrFsReason_t reason)
{
  eventTrigger_twinrFsTrigger_payload.reason = (uint8_t)reason;
  eventTrigger_twinrFsTrigger_payload.sessionId = twinrFsSessionId;
  eventTrigger_twinrFsTrigger_payload.vbatMv = twinrFsLastVbatMv;
  eventTrigger_twinrFsTrigger_payload.minClearanceMm = twinrFsLastMinClearanceMm;
  eventTrigger_twinrFsTrigger_payload.downRangeMm = twinrFsLastDownRangeMm;
  eventTrigger(&eventTrigger_twinrFsTrigger);
}

static void triggerFailsafe(const twinrFsReason_t reason,
                            const TickType_t now,
                            const float stateEstimateZ,
                            const uint16_t downMm)
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
  recordTriggerEvent(reason);
  setState(twinrFsStateFailsafeBrake, reason, now, true);
  DEBUG_PRINT("Failsafe triggered reason=%u targetZ=%.3f\n",
              (unsigned)reason,
              (double)twinrFsFailsafeTargetZ);
}

static void bindSession(const uint16_t sessionId)
{
  twinrFsSessionId = sessionId;
  twinrFsSessionBound = 1U;
  twinrFsHeartbeatSeqValid = 0U;
  twinrFsLastHeartbeatSeq = 0U;
}

static void applyHeartbeatConfig(const twinrFsHeartbeatConfig_t *config,
                                 const TickType_t now,
                                 const bool airborne)
{
  const bool airborneOrActive = airborne || twinrFsControlActive;

  if (!twinrFsSessionBound) {
    bindSession(config->sessionId);
  } else if (config->sessionId != twinrFsSessionId) {
    if (airborneOrActive) {
      noteReject(twinrFsRejectSessionMismatch,
                 config->version,
                 TWINR_FS_PACKET_KIND_HEARTBEAT,
                 config->sessionId,
                 0U);
      return;
    }

    bindSession(config->sessionId);
  }

  if (config->hasSequence) {
    if (twinrFsHeartbeatSeqValid && config->sequence <= twinrFsLastHeartbeatSeq) {
      noteReject(twinrFsRejectStaleSequence,
                 config->version,
                 TWINR_FS_PACKET_KIND_HEARTBEAT,
                 config->sessionId,
                 (uint16_t)(config->sequence & 0xffffU));
      return;
    }
    twinrFsHeartbeatSeqValid = 1U;
    twinrFsLastHeartbeatSeq = config->sequence;
  } else if (twinrFsHeartbeatSeqValid && airborneOrActive) {
    noteReject(twinrFsRejectDowngrade,
               config->version,
               TWINR_FS_PACKET_KIND_HEARTBEAT,
               config->sessionId,
               0U);
    return;
  }

  twinrFsLastHeartbeatTick = now;
  twinrFsHeartbeatAgeMs = 0U;
  twinrFsPacketVersionInUse = config->version;

  if (airborneOrActive) {
    if ((config->flags & TWINR_FS_FLAG_ENABLE) == 0U) {
      noteReject(twinrFsRejectInFlightDisable,
                 config->version,
                 TWINR_FS_PACKET_KIND_HEARTBEAT,
                 config->sessionId,
                 0U);
    } else if (!heartbeatConfigEqualsCurrent(config)) {
      noteReject(twinrFsRejectInFlightReconfigure,
                 config->version,
                 TWINR_FS_PACKET_KIND_HEARTBEAT,
                 config->sessionId,
                 0U);
    }
    return;
  }

  twinrFsEnable = (config->flags & TWINR_FS_FLAG_ENABLE) ? 1U : 0U;
  twinrFsRequireClearance = (config->flags & TWINR_FS_FLAG_REQUIRE_CLEARANCE) ? 1U : 0U;
  twinrFsHeartbeatTimeoutMs = config->heartbeatTimeoutMs;
  twinrFsLowBatteryMv = config->lowBatteryMv;
  twinrFsCriticalBatteryMv = config->criticalBatteryMv;
  twinrFsMinClearanceMm = config->minClearanceMm;
  twinrFsMinUpClearanceMm = config->minUpClearanceMm;
  twinrFsDescentRateMmps = config->descentRateMmps;
  twinrFsMaxRepelVelocityMmps = config->maxRepelVelocityMmps;
  twinrFsBrakeHoldMs = config->brakeHoldMs;
  twinrFsLowBatteryDebounceTicks = config->lowBatteryDebounceTicks;
  twinrFsCriticalBatteryDebounceTicks = config->criticalBatteryDebounceTicks;
  twinrFsClearanceDebounceTicks = config->clearanceDebounceTicks;
  resetTriggerCounters();

  if (!twinrFsEnable && !twinrFsControlActive) {
    setState(twinrFsStateDisabled, twinrFsReasonManualDisable, now, true);
  } else if (!twinrFsControlActive) {
    setState(twinrFsStateMonitoring, twinrFsReasonNone, now, true);
  }
}

static void processHeartbeatPackets(const TickType_t now, const bool airborne)
{
  uint8_t buffer[APPCHANNEL_MTU];
  size_t length = 0U;

  while ((length = appchannelReceiveDataPacket(buffer, sizeof(buffer), 0)) > 0U) {
    twinrFsHeartbeatConfig_t config;
    if (!parseHeartbeatPacket(buffer, length, &config)) {
      continue;
    }

    applyHeartbeatConfig(&config, now, airborne);
  }

  if (appchannelHasOverflowOccurred()) {
    if (twinrFsOverflowCount < UINT16_MAX) {
      twinrFsOverflowCount += 1U;
    }

    eventTrigger_twinrFsOverflow_payload.sessionId = twinrFsSessionId;
    eventTrigger_twinrFsOverflow_payload.heartbeatAgeMs = twinrFsHeartbeatAgeMs;
    eventTrigger(&eventTrigger_twinrFsOverflow);

    DEBUG_PRINT("Appchannel RX overflow\n");
  }
}

static void runFailsafeControl(const TickType_t now,
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
    const float descentPerStep = (((float)twinrFsDescentRateMmps) / 1000.0f)
                               * (((float)TWINR_FS_LOOP_PERIOD_MS) / 1000.0f);
    if (twinrFsFailsafeTargetZ > 0.0f) {
      twinrFsFailsafeTargetZ = fmaxf(0.0f, twinrFsFailsafeTargetZ - descentPerStep);
    }
    if (rangeIsValid(downMm) && downMm <= TWINR_FS_LANDING_FLOOR_MM) {
      twinrFsFailsafeTargetZ = fminf(twinrFsFailsafeTargetZ,
                                    ((float)TWINR_FS_LANDING_FLOOR_MM) / 1000.0f);
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
        eventTrigger_twinrFsLanded_payload.reason = twinrFsReason;
        eventTrigger_twinrFsLanded_payload.sessionId = twinrFsSessionId;
        eventTrigger_twinrFsLanded_payload.downRangeMm = downMm;
        eventTrigger(&eventTrigger_twinrFsLanded);
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
  twinrFsPmStateId = logGetVarId("pm", "state");
  twinrFsStateEstimateZId = logGetVarId("stateEstimate", "z");

  TickType_t lastWakeTime = xTaskGetTickCount();
  DEBUG_PRINT("Twinr on-device failsafe app ready\n");

  while (1) {
    vTaskDelayUntil(&lastWakeTime, M2T(TWINR_FS_LOOP_PERIOD_MS));
    const TickType_t now = xTaskGetTickCount();

    const uint16_t frontMm = readRangeMm(twinrFsFrontId);
    const uint16_t backMm = readRangeMm(twinrFsBackId);
    const uint16_t leftMm = readRangeMm(twinrFsLeftId);
    const uint16_t rightMm = readRangeMm(twinrFsRightId);
    const uint16_t upMm = readRangeMm(twinrFsUpId);
    const uint16_t downMm = readRangeMm(twinrFsDownId);
    const uint16_t vbatMv = readVbatMv();
    const uint8_t pmState = readPmState();
    const float stateEstimateZ = readStateEstimateZ();
    const bool isFlying = supervisorIsFlying();
    const bool airborne = isFlying
                       || (rangeIsValid(downMm) && downMm > TWINR_FS_LANDING_FLOOR_MM)
                       || stateEstimateZ > TWINR_FS_MIN_ACTIVE_ALTITUDE_M;

    processHeartbeatPackets(now, airborne);

    twinrFsLastVbatMv = vbatMv;
    twinrFsLastPmState = pmState;
    twinrFsLastMinClearanceMm = minClearanceMm(frontMm, backMm, leftMm, rightMm, upMm);
    twinrFsLastDownRangeMm = downMm;
    twinrFsHeartbeatAgeMs = twinrFsLastHeartbeatTick == 0
                              ? 0U
                              : clampUint16(T2M(now - twinrFsLastHeartbeatTick));

    if (twinrFsControlActive) {
      runFailsafeControl(now, stateEstimateZ, frontMm, backMm, leftMm, rightMm, downMm, isFlying);
      maybeSendStatus(now, false);
      continue;
    }

    if (!twinrFsEnable) {
      resetTriggerCounters();
      setState(twinrFsStateDisabled, twinrFsReasonManualDisable, now, false);
      maybeSendStatus(now, false);
      continue;
    }

    if (!airborne) {
      resetTriggerCounters();
      setState(twinrFsStateMonitoring, twinrFsReasonNone, now, false);
      maybeSendStatus(now, false);
      continue;
    }

    twinrFsReason_t triggerReason = twinrFsReasonNone;
    if (batteryViolated(vbatMv, pmState, &triggerReason)) {
      /* triggerReason already set */
    } else if (clearanceDebounced(frontMm, backMm, leftMm, rightMm, upMm, &triggerReason)) {
      /* triggerReason already set */
    } else if (heartbeatExpired(now)) {
      triggerReason = twinrFsReasonHeartbeatLoss;
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
LOG_ADD(LOG_UINT8, pmState, &twinrFsLastPmState)
LOG_ADD(LOG_UINT16, minClearanceMm, &twinrFsLastMinClearanceMm)
LOG_ADD(LOG_UINT16, downRangeMm, &twinrFsLastDownRangeMm)
LOG_ADD(LOG_UINT8, lastRejectCode, &twinrFsLastRejectCode)
LOG_ADD(LOG_UINT16, rejectedPkts, &twinrFsRejectedPacketCount)
LOG_ADD(LOG_UINT16, overflowCount, &twinrFsOverflowCount)
LOG_ADD(LOG_UINT32, heartbeatSeq, &twinrFsLastHeartbeatSeq)
LOG_GROUP_STOP(twinrFs)

PARAM_GROUP_START(twinrFs)
PARAM_ADD_CORE(PARAM_UINT8 | PARAM_RONLY, protocolVersion, &twinrFsProtocolVersion)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, packetVersionInUse, &twinrFsPacketVersionInUse)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, enable, &twinrFsEnable)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, requireClearance, &twinrFsRequireClearance)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, sessionId, &twinrFsSessionId)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, sessionBound, &twinrFsSessionBound)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, heartbeatTimeoutMs, &twinrFsHeartbeatTimeoutMs)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, lowBatteryMv, &twinrFsLowBatteryMv)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, criticalBatteryMv, &twinrFsCriticalBatteryMv)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, minClearanceMm, &twinrFsMinClearanceMm)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, minUpClearanceMm, &twinrFsMinUpClearanceMm)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, descentRateMmps, &twinrFsDescentRateMmps)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, maxRepelVelocityMmps, &twinrFsMaxRepelVelocityMmps)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, brakeHoldMs, &twinrFsBrakeHoldMs)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, lowBattDbTicks, &twinrFsLowBatteryDebounceTicks)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, critBattDbTicks, &twinrFsCriticalBatteryDebounceTicks)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, clrDbTicks, &twinrFsClearanceDebounceTicks)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, state, &twinrFsState)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, reason, &twinrFsReason)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, lastRejectCode, &twinrFsLastRejectCode)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, rejectedPkts, &twinrFsRejectedPacketCount)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, overflowCount, &twinrFsOverflowCount)
PARAM_GROUP_STOP(twinrFs)