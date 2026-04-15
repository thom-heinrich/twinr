#include "twinr_on_device_failsafe_internal.h"

/**
 * Mission control only orchestrates phases and maps explicit module decisions
 * into one live `twinrFs` control lane. Vertical estimation and lateral
 * disturbance observation now live in dedicated pure-C helpers.
 */

static float thrustRatioToCommand(const float ratio)
{
  const float boundedRatio = clampFloat(ratio, 0.0f, 1.0f);
  return boundedRatio * (float)UINT16_MAX;
}

static twinrFsReason_t verticalAbortReason(const twinrFsVerticalTakeoffDecision_t *decision)
{
  if (decision->progressClass == twinrFsVerticalProgressCeilingWithoutProgress) {
    return twinrFsReasonCeilingWithoutProgress;
  }
  if (decision->progressClass == twinrFsVerticalProgressOvershoot) {
    return twinrFsReasonTakeoffOvershoot;
  }
  return twinrFsReasonMissionAbort;
}

static void updateTakeoffDebugFlags(const bool rangeLive,
                                    const bool flowLive,
                                    const bool attitudeQuiet,
                                    const bool thrustAtCeiling,
                                    const bool disturbanceValid)
{
  uint8_t flags = 0U;
  if (rangeLive) {
    flags |= TWINR_FS_DEBUG_FLAG_RANGE_READY;
  }
  if (flowLive) {
    flags |= TWINR_FS_DEBUG_FLAG_FLOW_READY;
  }
  if (attitudeQuiet) {
    flags |= TWINR_FS_DEBUG_FLAG_ATTITUDE_READY;
  }
  if (thrustAtCeiling) {
    flags |= TWINR_FS_DEBUG_FLAG_THRUST_AT_CEILING;
  }
  if (twinrFsHoverThrustEstimate > 0.0f) {
    flags |= TWINR_FS_DEBUG_FLAG_HOVER_THRUST_VALID;
  }
  if (disturbanceValid) {
    flags |= TWINR_FS_DEBUG_FLAG_DISTURBANCE_VALID;
  }
  twinrFsTakeoffDebugFlags = flags;
}

void runMissionControl(const TickType_t now,
                       const float stateEstimateZ,
                       const uint16_t downMm,
                       const uint16_t motionSqual,
                       const bool isFlying)
{
  (void)motionSqual;
  const twinrFsVerticalControlConfig_t verticalConfig = twinrFsVerticalDefaultConfig();
  const twinrFsDisturbanceControlConfig_t disturbanceConfig = twinrFsDisturbanceDefaultConfig();
  const bool airborne = flightObserved(isFlying, downMm, stateEstimateZ);
  const bool rangeLive = missionRangeLive();
  const bool flowLive = missionFlowLive();
  const bool attitudeQuiet = missionAttitudeQuiet();
  const bool truthStale = missionTruthStale();
  const bool stateFlapping = missionStateFlapping();

  if (!twinrFsMissionActive) {
    return;
  }

  if (twinrFsState == twinrFsStateMissionTakeoff) {
    const uint32_t phaseElapsedMs = T2M(now - twinrFsMissionPhaseStartTick);
    const twinrFsVerticalTakeoffDecision_t verticalDecision = twinrFsVerticalControlStepTakeoff(
        &twinrFsVerticalControl,
        &verticalConfig,
        &(const twinrFsVerticalTakeoffObservation_t){
            .vbatMv = twinrFsLastVbatMv,
            .downMm = downMm,
            .baselineHeightMm = twinrFsMissionBaselineHeightMm,
            .microHeightMm = twinrFsMissionMicroHeightMm,
            .targetToleranceMm = twinrFsMissionTargetToleranceMm,
            .takeoffRampMs = twinrFsMissionTakeoffRampMs,
            .loopPeriodMs = TWINR_FS_LOOP_PERIOD_MS,
            .phaseElapsedMs = phaseElapsedMs,
            .stateEstimateVz = twinrFsLastStateEstimateVz,
            .rangeLive = rangeLive,
            .flowLive = flowLive,
            .attitudeQuiet = attitudeQuiet,
        });
    updateTakeoffDebugFlags(
        rangeLive,
        flowLive,
        attitudeQuiet,
        verticalDecision.thrustAtCeiling,
        false);
    twinrFsMissionCommandedHeightMm = verticalDecision.commandedHeightMm;
    sendManualTakeoffSetpoint(
        thrustRatioToCommand(verticalDecision.thrustRatio),
        twinrFsLateralCommandSourceMissionTakeoff);

    if (verticalDecision.shouldAbort) {
      triggerFailsafe(
          verticalAbortReason(&verticalDecision),
          now,
          stateEstimateZ,
          downMm);
      return;
    }

    if (verticalDecision.shouldHandoff) {
      twinrFsMissionTakeoffProven = 1U;
      twinrFsMissionHoverQualified = 1U;
      twinrFsMissionCommandedHeightMm = twinrFsMissionTargetHeightMm;
      twinrFsMissionPhaseStartTick = now;
      setState(
          twinrFsStateMissionHover,
          twinrFsReasonNone,
          now,
          true,
          airborne,
          rangeLive,
          flowLive,
          stateEstimateZ);
      return;
    }

    if (phaseElapsedMs >= TWINR_FS_HOVER_TAKEOFF_TIMEOUT_MS) {
      twinrFsReason_t takeoffReason = twinrFsReasonTakeoffRangeLiveness;
      if (stateFlapping) {
        takeoffReason = twinrFsReasonStateFlapping;
      } else if (verticalDecision.progressClass == twinrFsVerticalProgressCeilingWithoutProgress) {
        takeoffReason = twinrFsReasonCeilingWithoutProgress;
      } else if (verticalDecision.progressClass == twinrFsVerticalProgressOvershoot) {
        takeoffReason = twinrFsReasonTakeoffOvershoot;
      } else if (rangeLive && !flowLive) {
        takeoffReason = twinrFsReasonTakeoffFlowLiveness;
      } else if (rangeLive && flowLive && !attitudeQuiet) {
        takeoffReason = twinrFsReasonTakeoffAttitudeQuiet;
      }
      triggerFailsafe(takeoffReason, now, stateEstimateZ, downMm);
      return;
    }

    maybeSendStatus(now, false, airborne, rangeLive, flowLive, stateEstimateZ);
    return;
  }

  if (twinrFsState == twinrFsStateMissionHover) {
    if (truthStale) {
      triggerFailsafe(twinrFsReasonTruthStale, now, stateEstimateZ, downMm);
      return;
    }

    twinrFsVerticalControlObserveHover(
        &twinrFsVerticalControl,
        &verticalConfig,
        &(const twinrFsVerticalHoverObservation_t){
            .stateEstimateZ = stateEstimateZ,
            .stateEstimateVz = twinrFsLastStateEstimateVz,
            .observedThrust = (float)twinrFsLastObservedThrust,
            .vbatMv = twinrFsLastVbatMv,
        },
        ((float)twinrFsMissionTargetHeightMm) / 1000.0f);
    const twinrFsDisturbanceDecision_t disturbanceDecision = twinrFsDisturbanceControlStep(
        &twinrFsDisturbanceControl,
        &disturbanceConfig,
        &(const twinrFsDisturbanceObservation_t){
            .flowLive = flowLive,
            .downMm = downMm,
            .stateEstimateVx = twinrFsLastStateEstimateVx,
            .stateEstimateVy = twinrFsLastStateEstimateVy,
        });
    updateTakeoffDebugFlags(rangeLive, flowLive, attitudeQuiet, false, disturbanceDecision.valid);
    if (disturbanceDecision.shouldAbort) {
      triggerFailsafe(twinrFsReasonDisturbanceNonrecoverable, now, stateEstimateZ, downMm);
      return;
    }

    twinrFsMissionTakeoffProven = 1U;
    twinrFsMissionHoverQualified = 1U;
    twinrFsMissionCommandedHeightMm = twinrFsMissionTargetHeightMm;
    sendFailsafeSetpoint(
        disturbanceDecision.vxCommand,
        disturbanceDecision.vyCommand,
        ((float)twinrFsMissionTargetHeightMm) / 1000.0f,
        twinrFsLateralCommandSourceMissionHover);
    if (twinrFsMissionLandRequested
        || T2M(now - twinrFsMissionPhaseStartTick) >= twinrFsMissionHoverDurationMs) {
      twinrFsMissionPhaseStartTick = now;
      twinrFsMissionTouchdownConfirmCount = 0U;
      setState(
          twinrFsStateMissionLanding,
          twinrFsReasonNone,
          now,
          true,
          airborne,
          rangeLive,
          flowLive,
          stateEstimateZ);
      return;
    }
    maybeSendStatus(now, false, airborne, rangeLive, flowLive, stateEstimateZ);
    return;
  }

  if (twinrFsState == twinrFsStateMissionLanding) {
    const uint16_t descentPerStepMm =
        (uint16_t)((((uint32_t)twinrFsDescentRateMmps) * TWINR_FS_LOOP_PERIOD_MS) / 1000U);
    const bool touchdownByRange = rangeIsValid(downMm) && downMm <= TWINR_FS_TOUCHDOWN_CONFIRM_MM;
    const bool touchdownBySupervisor = !isFlying;

    twinrFsTakeoffDebugFlags &= (uint8_t)~(TWINR_FS_DEBUG_FLAG_TOUCHDOWN_BY_RANGE
                                           | TWINR_FS_DEBUG_FLAG_TOUCHDOWN_BY_SUPERVISOR);
    if (touchdownByRange) {
      twinrFsTakeoffDebugFlags |= TWINR_FS_DEBUG_FLAG_TOUCHDOWN_BY_RANGE;
    }
    if (touchdownBySupervisor) {
      twinrFsTakeoffDebugFlags |= TWINR_FS_DEBUG_FLAG_TOUCHDOWN_BY_SUPERVISOR;
    }

    if (twinrFsMissionCommandedHeightMm > descentPerStepMm) {
      twinrFsMissionCommandedHeightMm = (uint16_t)(twinrFsMissionCommandedHeightMm - descentPerStepMm);
    } else {
      twinrFsMissionCommandedHeightMm = 0U;
    }
    sendFailsafeSetpoint(
        0.0f,
        0.0f,
        ((float)twinrFsMissionCommandedHeightMm) / 1000.0f,
        twinrFsLateralCommandSourceMissionLanding);

    if (touchdownByRange || touchdownBySupervisor) {
      if (twinrFsMissionTouchdownConfirmCount < UINT8_MAX) {
        twinrFsMissionTouchdownConfirmCount += 1U;
      }
    } else {
      twinrFsMissionTouchdownConfirmCount = 0U;
    }

    if (twinrFsMissionTouchdownConfirmCount >= TWINR_FS_TOUCHDOWN_CONFIRM_SAMPLES) {
      commanderRelaxPriority();
      supervisorRequestArming(false);
      twinrFsMissionActive = 0U;
      twinrFsMissionLandRequested = 0U;
      twinrFsMissionTakeoffProven = 0U;
      twinrFsMissionHoverQualified = 0U;
      setState(
          twinrFsStateMissionComplete,
          twinrFsReasonNone,
          now,
          true,
          false,
          rangeLive,
          flowLive,
          stateEstimateZ);
      return;
    }
    maybeSendStatus(now, false, airborne, rangeLive, flowLive, stateEstimateZ);
    return;
  }

  maybeSendStatus(now, false, airborne, rangeLive, flowLive, stateEstimateZ);
}
