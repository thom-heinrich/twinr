// # CHANGELOG: 2026-04-14
// # BUG-6: Active bounded-hover missions are now fully config-frozen from the first mission-start packet onward; heartbeat reconfiguration no longer mutates the takeoff lane before airborne proof.
// # BUG-7: Mission landing now uses the same multi-sample touchdown confirmation contract as the failsafe lane and can no longer disarm on a single transient landed sample.
// # IMP-3: The mission-takeoff lane now performs bounded on-device hover-thrust adaptation and disturbance estimation on the STM32 before handing off into qualified hover.
// # IMP-4: `twinrFs` now logs the last commanded lateral velocities plus their command-source lane so failed takeoff drift can be proven against the on-device command contract instead of inferred from motion alone.
// # ARCH-2: Split the STM32 twinrFs app into protocol, telemetry, mission-control, and failsafe-control modules while keeping one live control lane.
// # ARCH-3: The vertical takeoff/hover estimation and lateral disturbance observer now live in pure C helper modules so the mission state machine only orchestrates one control lane.

/**
 * Twinr on-device hover/failsafe app.
 *
 * The host is limited to mission intent and telemetry. This STM32 app owns the
 * live bounded-hover state machine, heartbeat safety contract, local landing
 * behavior, and the first on-device adaptive takeoff/hover estimation layer.
 */

#include "twinr_on_device_failsafe_internal.h"

twinrFsContext_t twinrFs = {
  .protocolVersion = TWINR_FS_PROTOCOL_VERSION,
  .requireClearance = 1U,
  .armLateralClearance = 1U,
  .packetVersionInUse = TWINR_FS_PROTOCOL_VERSION_LEGACY,
  .heartbeatTimeoutMs = TWINR_FS_HEARTBEAT_TIMEOUT_MS,
  .lowBatteryMv = TWINR_FS_LOW_BATTERY_MV,
  .criticalBatteryMv = TWINR_FS_CRITICAL_BATTERY_MV,
  .minClearanceMm = TWINR_FS_MIN_CLEARANCE_MM,
  .minUpClearanceMm = TWINR_FS_MIN_UP_CLEARANCE_MM,
  .descentRateMmps = TWINR_FS_DESCENT_RATE_MMPS,
  .maxRepelVelocityMmps = TWINR_FS_MAX_REPEL_VELOCITY_MMPS,
  .brakeHoldMs = TWINR_FS_BRAKE_HOLD_MS,
  .lowBatteryDebounceTicks = TWINR_FS_LOW_BATTERY_DEBOUNCE_TICKS_DEFAULT,
  .criticalBatteryDebounceTicks = TWINR_FS_CRITICAL_BATTERY_DEBOUNCE_TICKS_DEFAULT,
  .clearanceDebounceTicks = TWINR_FS_CLEARANCE_DEBOUNCE_TICKS_DEFAULT,
  .state = twinrFsStateDisabled,
  .reason = twinrFsReasonNone,
  .missionTargetHeightMm = TWINR_FS_HOVER_TARGET_HEIGHT_MM_DEFAULT,
  .missionMicroHeightMm = TWINR_FS_HOVER_MICRO_HEIGHT_MM_DEFAULT,
  .missionTargetToleranceMm = TWINR_FS_HOVER_TARGET_TOLERANCE_MM_DEFAULT,
  .missionHoverDurationMs = TWINR_FS_HOVER_DURATION_MS_DEFAULT,
  .missionTakeoffRampMs = TWINR_FS_HOVER_TAKEOFF_RAMP_MS_DEFAULT,
  .verticalControl =
      {
          .hoverThrustEstimate = TWINR_FS_TAKEOFF_THRUST_ESTIMATE_DEFAULT,
          .filteredBatteryMv = 4200.0f,
          .progressClass = twinrFsVerticalProgressIdle,
      },
  .disturbanceControl =
      {
          .recoverable = 1U,
          .severityClass = twinrFsDisturbanceSeverityNone,
      },
  .lateralCommandSource = twinrFsLateralCommandSourceNone,
};

static uint16_t clampObservedThrust(const float thrust)
{
  if (!isfinite(thrust) || thrust <= 0.0f) {
    return 0U;
  }
  if (thrust >= (float)UINT16_MAX) {
    return UINT16_MAX;
  }
  return (uint16_t)lrintf(thrust);
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
  twinrFsStateEstimateVxId = logGetVarId("stateEstimate", "vx");
  twinrFsStateEstimateVyId = logGetVarId("stateEstimate", "vy");
  twinrFsStateEstimateVzId = logGetVarId("stateEstimate", "vz");
  twinrFsMotionSqualId = logGetVarId("motion", "squal");
  twinrFsStabilizerRollId = logGetVarId("stabilizer", "roll");
  twinrFsStabilizerPitchId = logGetVarId("stabilizer", "pitch");
  twinrFsStabilizerThrustId = logGetVarId("stabilizer", "thrust");

  TickType_t lastWakeTime = xTaskGetTickCount();
  DEBUG_PRINT("Twinr on-device failsafe app ready\n");

  while (1) {
    vTaskDelayUntil(&lastWakeTime, M2T(TWINR_FS_LOOP_PERIOD_MS));
    const TickType_t now = xTaskGetTickCount();

    twinrFsObservation_t observation = {
      .frontMm = readRangeMm(twinrFsFrontId),
      .backMm = readRangeMm(twinrFsBackId),
      .leftMm = readRangeMm(twinrFsLeftId),
      .rightMm = readRangeMm(twinrFsRightId),
      .upMm = readRangeMm(twinrFsUpId),
      .downMm = readRangeMm(twinrFsDownId),
      .vbatMv = readVbatMv(),
      .motionSqual = readMotionSqual(),
      .pmState = readPmState(),
      .isFlying = supervisorIsFlying(),
      .stateEstimateZ = readStateEstimateZ(),
    };

    twinrFsLastStateEstimateVx = readStateEstimateVelocity(twinrFsStateEstimateVxId);
    twinrFsLastStateEstimateVy = readStateEstimateVelocity(twinrFsStateEstimateVyId);
    twinrFsLastStateEstimateVz = readStateEstimateVelocity(twinrFsStateEstimateVzId);
    twinrFsLastRollDeg = readStabilizerScalar(twinrFsStabilizerRollId);
    twinrFsLastPitchDeg = readStabilizerScalar(twinrFsStabilizerPitchId);
    twinrFsLastObservedThrust = clampObservedThrust(readStabilizerScalar(twinrFsStabilizerThrustId));

    runTwinrFsStateMachine(now, &observation);
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
LOG_ADD(LOG_UINT16, upMm, &twinrFsLastUpRangeMm)
LOG_ADD(LOG_UINT16, frontMm, &twinrFsLastFrontMm)
LOG_ADD(LOG_UINT16, backMm, &twinrFsLastBackMm)
LOG_ADD(LOG_UINT16, leftMm, &twinrFsLastLeftMm)
LOG_ADD(LOG_UINT16, rightMm, &twinrFsLastRightMm)
LOG_ADD(LOG_UINT16, mSqual, &twinrFsLastMotionSqual)
LOG_ADD(LOG_FLOAT, vx, &twinrFsLastStateEstimateVx)
LOG_ADD(LOG_FLOAT, vy, &twinrFsLastStateEstimateVy)
LOG_ADD(LOG_FLOAT, vz, &twinrFsLastStateEstimateVz)
LOG_ADD(LOG_FLOAT, roll, &twinrFsLastRollDeg)
LOG_ADD(LOG_FLOAT, pitch, &twinrFsLastPitchDeg)
LOG_ADD(LOG_FLOAT, cmdVx, &twinrFsLastCommandedVx)
LOG_ADD(LOG_FLOAT, cmdVy, &twinrFsLastCommandedVy)
LOG_ADD(LOG_UINT8, cmdSrc, &twinrFsLateralCommandSource)
LOG_ADD(LOG_UINT8, tkDbg, &twinrFsTakeoffDebugFlags)
LOG_ADD(LOG_UINT8, tkRfCnt, &twinrFsMissionRangeFreshCount)
LOG_ADD(LOG_UINT8, tkRrCnt, &twinrFsMissionRangeRiseCount)
LOG_ADD(LOG_UINT8, tkFlCnt, &twinrFsMissionFlowReadyCount)
LOG_ADD(LOG_UINT8, tkAtCnt, &twinrFsMissionAttitudeQuietCount)
LOG_ADD(LOG_UINT8, tkStCnt, &twinrFsMissionTruthStaleCount)
LOG_ADD(LOG_UINT8, tkFpCnt, &twinrFsMissionTruthFlapCount)
LOG_ADD(LOG_UINT8, tkPgCls, &twinrFsTakeoffProgressClass)
LOG_ADD(LOG_FLOAT, tkBatMv, &twinrFsFilteredBatteryMv)
LOG_ADD(LOG_UINT8, tdCnt, &twinrFsTouchdownConfirmCount)
LOG_ADD(LOG_UINT8, mTdCnt, &twinrFsMissionTouchdownConfirmCount)
LOG_ADD(LOG_FLOAT, thrEst, &twinrFsHoverThrustEstimate)
LOG_ADD(LOG_FLOAT, distVx, &twinrFsDisturbanceEstimateVx)
LOG_ADD(LOG_FLOAT, distVy, &twinrFsDisturbanceEstimateVy)
LOG_ADD(LOG_UINT16, distSev, &twinrFsDisturbanceSeverityPermille)
LOG_ADD(LOG_UINT8, distRec, &twinrFsDisturbanceRecoverable)
LOG_ADD(LOG_UINT8, lastRejectCode, &twinrFsLastRejectCode)
LOG_ADD(LOG_UINT16, rejectedPkts, &twinrFsRejectedPacketCount)
LOG_ADD(LOG_UINT16, overflowCount, &twinrFsOverflowCount)
LOG_ADD(LOG_UINT32, heartbeatSeq, &twinrFsLastHeartbeatSeq)
LOG_GROUP_STOP(twinrFs)

PARAM_GROUP_START(twinrFs)
// Bitcraze CMD_GET_ITEM_V2 caps group+name+overhead at 26 bytes.
PARAM_ADD_CORE(PARAM_UINT8 | PARAM_RONLY, protocolVersion, &twinrFsProtocolVersion)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, pktVersionUse, &twinrFsPacketVersionInUse)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, enable, &twinrFsEnable)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, requireClearance, &twinrFsRequireClearance)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, lateralClrArm, &twinrFsArmLateralClearance)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, sessionId, &twinrFsSessionId)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, sessionBound, &twinrFsSessionBound)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, heartbeatToutMs, &twinrFsHeartbeatTimeoutMs)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, lowBatteryMv, &twinrFsLowBatteryMv)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, criticalBatteryMv, &twinrFsCriticalBatteryMv)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, minClearanceMm, &twinrFsMinClearanceMm)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, minUpClearanceMm, &twinrFsMinUpClearanceMm)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, descentRateMmps, &twinrFsDescentRateMmps)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, maxRepelVelMmps, &twinrFsMaxRepelVelocityMmps)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, brakeHoldMs, &twinrFsBrakeHoldMs)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, lowBattDbTicks, &twinrFsLowBatteryDebounceTicks)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, critBattDbTicks, &twinrFsCriticalBatteryDebounceTicks)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, clrDbTicks, &twinrFsClearanceDebounceTicks)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, state, &twinrFsState)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, reason, &twinrFsReason)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, tkDbg, &twinrFsTakeoffDebugFlags)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, lastRejectCode, &twinrFsLastRejectCode)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, rejectedPkts, &twinrFsRejectedPacketCount)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, overflowCount, &twinrFsOverflowCount)
PARAM_GROUP_STOP(twinrFs)
