#include "twinr_on_device_failsafe_internal.h"

EVENTTRIGGER(twinrFsTrigger,
             uint8, reason,
             uint16, sessionId,
             uint16, vbatMv,
             uint16, minClearanceMm,
             uint16, downRangeMm)

EVENTTRIGGER(twinrFsLanded,
             uint8, reason,
             uint16, sessionId,
             uint16, downRangeMm)

uint16_t clampUint16(const uint32_t value)
{
  return value > UINT16_MAX ? UINT16_MAX : (uint16_t)value;
}

uint16_t clampUint16Range(const uint16_t value, const uint16_t lower, const uint16_t upper)
{
  if (value < lower) {
    return lower;
  }
  if (value > upper) {
    return upper;
  }
  return value;
}

uint8_t clampUint8Range(const uint8_t value, const uint8_t lower, const uint8_t upper)
{
  if (value < lower) {
    return lower;
  }
  if (value > upper) {
    return upper;
  }
  return value;
}

float clampFloat(const float value, const float lower, const float upper)
{
  if (value < lower) {
    return lower;
  }
  if (value > upper) {
    return upper;
  }
  return value;
}

bool rangeIsValid(const uint16_t rangeMm)
{
  return rangeMm > 0U && rangeMm < TWINR_FS_RANGE_INVALID_MM;
}

uint16_t readRangeMm(const logVarId_t id)
{
  if (!logVarIdIsValid(id)) {
    return 0U;
  }
  return clampUint16(logGetUint(id));
}

uint16_t readVbatMv(void)
{
  if (!logVarIdIsValid(twinrFsVbatMvId)) {
    return 0U;
  }
  return clampUint16(logGetUint(twinrFsVbatMvId));
}

uint8_t readPmState(void)
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

float readStateEstimateZ(void)
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

float readStateEstimateVelocity(const logVarId_t id)
{
  if (!logVarIdIsValid(id)) {
    return 0.0f;
  }
  const float value = logGetFloat(id);
  return isfinite(value) ? value : 0.0f;
}

float readStabilizerScalar(const logVarId_t id)
{
  if (!logVarIdIsValid(id)) {
    return 0.0f;
  }
  const float value = logGetFloat(id);
  return isfinite(value) ? value : 0.0f;
}

uint16_t readMotionSqual(void)
{
  if (!logVarIdIsValid(twinrFsMotionSqualId)) {
    return 0U;
  }
  return clampUint16(logGetUint(twinrFsMotionSqualId));
}

void resetTriggerCounters(void)
{
  twinrFsLowBatteryCount = 0U;
  twinrFsCriticalBatteryCount = 0U;
  twinrFsClearanceCount = 0U;
  twinrFsPendingClearanceReason = twinrFsReasonNone;
}

void resetBatteryCounters(void)
{
  twinrFsLowBatteryCount = 0U;
  twinrFsCriticalBatteryCount = 0U;
}

void resetMissionState(void)
{
  const twinrFsVerticalControlConfig_t verticalConfig = twinrFsVerticalDefaultConfig();
  twinrFsMissionActive = 0U;
  twinrFsMissionLandRequested = 0U;
  twinrFsMissionTakeoffProven = 0U;
  twinrFsMissionHoverQualified = 0U;
  twinrFsMissionBaselineHeightMm = 0U;
  twinrFsMissionCommandedHeightMm = 0U;
  twinrFsMissionStartTick = 0;
  twinrFsMissionPhaseStartTick = 0;
  twinrFsMissionTouchdownConfirmCount = 0U;
  twinrFsMissionRangeFreshCount = 0U;
  twinrFsMissionRangeRiseCount = 0U;
  twinrFsMissionFlowReadyCount = 0U;
  twinrFsMissionAttitudeQuietCount = 0U;
  twinrFsMissionTruthStaleCount = 0U;
  twinrFsMissionTruthFlapCount = 0U;
  twinrFsMissionRangeFreshRaw = 0U;
  twinrFsMissionRangeRiseRaw = 0U;
  twinrFsMissionFlowReadyRaw = 0U;
  twinrFsMissionAttitudeQuietRaw = 0U;
  twinrFsTakeoffDebugFlags = 0U;
  twinrFsLastCommandedVx = 0.0f;
  twinrFsLastCommandedVy = 0.0f;
  twinrFsLateralCommandSource = twinrFsLateralCommandSourceNone;
  twinrFsVerticalControlReset(&twinrFsVerticalControl, &verticalConfig);
  twinrFsDisturbanceControlReset(&twinrFsDisturbanceControl);
}

uint16_t stateEstimateZToMm(const float stateEstimateZ)
{
  if (!isfinite(stateEstimateZ) || stateEstimateZ <= 0.0f) {
    return 0U;
  }
  return clampUint16((uint32_t)lrintf(stateEstimateZ * 1000.0f));
}

uint8_t currentMissionFlags(const bool airborne,
                            const bool rangeLive,
                            const bool flowLive)
{
  uint8_t flags = 0U;
  if (airborne) {
    flags |= TWINR_FS_HOVER_STATE_FLAG_AIRBORNE;
  }
  if (rangeLive) {
    flags |= TWINR_FS_HOVER_STATE_FLAG_RANGE_LIVE;
  }
  if (flowLive) {
    flags |= TWINR_FS_HOVER_STATE_FLAG_FLOW_LIVE;
  }
  if (twinrFsMissionActive) {
    flags |= TWINR_FS_HOVER_STATE_FLAG_MISSION_ACTIVE;
  }
  if (twinrFsMissionTakeoffProven) {
    flags |= TWINR_FS_HOVER_STATE_FLAG_TAKEOFF_PROVEN;
  }
  if (twinrFsMissionHoverQualified) {
    flags |= TWINR_FS_HOVER_STATE_FLAG_HOVER_QUALIFIED;
  }
  if (twinrFsState == twinrFsStateMissionLanding) {
    flags |= TWINR_FS_HOVER_STATE_FLAG_LANDING_ACTIVE;
  }
  if (twinrFsState == twinrFsStateMissionComplete || twinrFsState == twinrFsStateLanded) {
    flags |= TWINR_FS_HOVER_STATE_FLAG_COMPLETE;
  }
  return flags;
}

static uint8_t nextMissionCount(const bool condition, const uint8_t current)
{
  if (!condition) {
    return 0U;
  }
  if (current >= UINT8_MAX) {
    return UINT8_MAX;
  }
  return (uint8_t)(current + 1U);
}

static uint8_t incrementMissionCount(const bool condition, const uint8_t current)
{
  if (!condition) {
    return current;
  }
  if (current >= UINT8_MAX) {
    return UINT8_MAX;
  }
  return (uint8_t)(current + 1U);
}

void updateMissionTruthCounters(const uint16_t downMm, const uint16_t motionSqual)
{
  /* Takeoff proof must survive more than one loop tick before hover handoff. */
  const bool rangeFresh = rangeIsValid(downMm);
  const uint16_t requiredRiseMm =
      (uint16_t)(twinrFsMissionBaselineHeightMm + TWINR_FS_HOVER_MIN_RANGE_RISE_MM);
  const bool rangeRiseReady = rangeFresh
                              && downMm >= twinrFsMissionMicroHeightMm
                              && downMm >= requiredRiseMm;
  const bool flowReady = rangeFresh
                         && downMm >= TWINR_FS_HOVER_FLOW_GATE_MIN_HEIGHT_MM
                         && motionSqual >= TWINR_FS_HOVER_FLOW_MIN_SQUAL;
  const bool attitudeQuiet = fabsf(twinrFsLastRollDeg) <= TWINR_FS_HOVER_ATTITUDE_QUIET_MAX_DEG
                             && fabsf(twinrFsLastPitchDeg) <= TWINR_FS_HOVER_ATTITUDE_QUIET_MAX_DEG;
  const bool truthRegressed = (twinrFsMissionRangeFreshRaw != 0U && !rangeFresh)
                              || (twinrFsMissionRangeRiseRaw != 0U && !rangeRiseReady)
                              || (twinrFsMissionFlowReadyRaw != 0U && !flowReady)
                              || (twinrFsMissionAttitudeQuietRaw != 0U && !attitudeQuiet);
  const bool hoverTruthMissing = twinrFsState == twinrFsStateMissionHover && (!rangeFresh || !flowReady);

  twinrFsMissionRangeFreshCount = nextMissionCount(rangeFresh, twinrFsMissionRangeFreshCount);
  twinrFsMissionRangeRiseCount = nextMissionCount(rangeRiseReady, twinrFsMissionRangeRiseCount);
  twinrFsMissionFlowReadyCount = nextMissionCount(flowReady, twinrFsMissionFlowReadyCount);
  twinrFsMissionAttitudeQuietCount = nextMissionCount(attitudeQuiet, twinrFsMissionAttitudeQuietCount);
  twinrFsMissionTruthStaleCount = nextMissionCount(hoverTruthMissing, twinrFsMissionTruthStaleCount);
  twinrFsMissionTruthFlapCount = incrementMissionCount(truthRegressed, twinrFsMissionTruthFlapCount);
  twinrFsMissionRangeFreshRaw = rangeFresh ? 1U : 0U;
  twinrFsMissionRangeRiseRaw = rangeRiseReady ? 1U : 0U;
  twinrFsMissionFlowReadyRaw = flowReady ? 1U : 0U;
  twinrFsMissionAttitudeQuietRaw = attitudeQuiet ? 1U : 0U;
}

bool missionRangeLive(void)
{
  return twinrFsMissionRangeFreshCount >= TWINR_FS_HOVER_RANGE_FRESH_SAMPLES
         && twinrFsMissionRangeRiseCount >= TWINR_FS_HOVER_RANGE_RISE_SAMPLES;
}

bool missionFlowLive(void)
{
  return twinrFsMissionFlowReadyCount >= TWINR_FS_HOVER_FLOW_LIVE_SAMPLES;
}

bool missionAttitudeQuiet(void)
{
  return twinrFsMissionAttitudeQuietCount >= TWINR_FS_HOVER_ATTITUDE_QUIET_SAMPLES;
}

bool missionTruthStale(void)
{
  return twinrFsMissionTruthStaleCount >= TWINR_FS_HOVER_TRUTH_STALE_SAMPLES;
}

bool missionStateFlapping(void)
{
  return twinrFsMissionTruthFlapCount >= TWINR_FS_HOVER_TRUTH_FLAP_LIMIT;
}

bool flightObserved(const bool isFlying,
                    const uint16_t downMm,
                    const float stateEstimateZ)
{
  if (rangeIsValid(downMm)) {
    return downMm > TWINR_FS_LANDING_FLOOR_MM;
  }
  return isFlying && stateEstimateZ > TWINR_FS_MIN_ACTIVE_ALTITUDE_M;
}

void bindSession(const uint16_t sessionId)
{
  const twinrFsVerticalControlConfig_t verticalConfig = twinrFsVerticalDefaultConfig();
  twinrFsSessionId = sessionId;
  twinrFsSessionBound = 1U;
  twinrFsHeartbeatSeqValid = 0U;
  twinrFsLastHeartbeatSeq = 0U;
  twinrFsFlightObserved = 0U;
  twinrFsGroundAbortQuiet = 0U;
  twinrFsTouchdownConfirmCount = 0U;
  twinrFsMissionTouchdownConfirmCount = 0U;
  twinrFsMissionRangeFreshCount = 0U;
  twinrFsMissionRangeRiseCount = 0U;
  twinrFsMissionFlowReadyCount = 0U;
  twinrFsMissionAttitudeQuietCount = 0U;
  twinrFsMissionTruthStaleCount = 0U;
  twinrFsMissionTruthFlapCount = 0U;
  twinrFsMissionRangeFreshRaw = 0U;
  twinrFsMissionRangeRiseRaw = 0U;
  twinrFsMissionFlowReadyRaw = 0U;
  twinrFsMissionAttitudeQuietRaw = 0U;
  twinrFsTakeoffDebugFlags = 0U;
  twinrFsLastCommandedVx = 0.0f;
  twinrFsLastCommandedVy = 0.0f;
  twinrFsLateralCommandSource = twinrFsLateralCommandSourceNone;
  twinrFsVerticalControlReset(&twinrFsVerticalControl, &verticalConfig);
  twinrFsDisturbanceControlReset(&twinrFsDisturbanceControl);
}

static uint16_t hoverThrustEstimatePermille(void)
{
  const float estimate = clampFloat(twinrFsHoverThrustEstimate, 0.0f, 1.0f);
  return clampUint16((uint32_t)lrintf(estimate * 1000.0f));
}

static uint8_t touchdownConfirmCount(void)
{
  if (twinrFsState == twinrFsStateMissionLanding) {
    return twinrFsMissionTouchdownConfirmCount;
  }
  return twinrFsTouchdownConfirmCount;
}

void maybeSendStatus(const TickType_t now,
                     const bool force,
                     const bool airborne,
                     const bool rangeLive,
                     const bool flowLive,
                     const float stateEstimateZ)
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
    .missionFlags = currentMissionFlags(airborne, rangeLive, flowLive),
    .debugFlags = twinrFsTakeoffDebugFlags,
    .targetHeightMm = twinrFsMissionTargetHeightMm,
    .commandedHeightMm = twinrFsMissionCommandedHeightMm,
    .stateEstimateZMm = stateEstimateZToMm(stateEstimateZ),
    .upRangeMm = twinrFsLastUpRangeMm,
    .motionSqual = twinrFsLastMotionSqual,
    .touchdownConfirmCount = touchdownConfirmCount(),
    .reserved = 0U,
    .hoverThrustPermille = hoverThrustEstimatePermille(),
  };

  appchannelSendDataPacket(&packet, sizeof(packet));
  twinrFsLastStatusTick = now;
}

void setState(const twinrFsState_t state,
              const twinrFsReason_t reason,
              const TickType_t now,
              const bool forceStatus,
              const bool airborne,
              const bool rangeLive,
              const bool flowLive,
              const float stateEstimateZ)
{
  const bool changed = (twinrFsState != (uint8_t)state) || (twinrFsReason != (uint8_t)reason);
  twinrFsState = (uint8_t)state;
  twinrFsReason = (uint8_t)reason;
  if (changed) {
    twinrFsLastStateTick = now;
  }
  maybeSendStatus(now, forceStatus || changed, airborne, rangeLive, flowLive, stateEstimateZ);
}

void recordTriggerEvent(const twinrFsReason_t reason)
{
  eventTrigger_twinrFsTrigger_payload.reason = (uint8_t)reason;
  eventTrigger_twinrFsTrigger_payload.sessionId = twinrFsSessionId;
  eventTrigger_twinrFsTrigger_payload.vbatMv = twinrFsLastVbatMv;
  eventTrigger_twinrFsTrigger_payload.minClearanceMm = twinrFsLastMinClearanceMm;
  eventTrigger_twinrFsTrigger_payload.downRangeMm = twinrFsLastDownRangeMm;
  eventTrigger(&eventTrigger_twinrFsTrigger);
}

void recordLandedEvent(const uint16_t downMm)
{
  eventTrigger_twinrFsLanded_payload.reason = twinrFsReason;
  eventTrigger_twinrFsLanded_payload.sessionId = twinrFsSessionId;
  eventTrigger_twinrFsLanded_payload.downRangeMm = downMm;
  eventTrigger(&eventTrigger_twinrFsLanded);
}
