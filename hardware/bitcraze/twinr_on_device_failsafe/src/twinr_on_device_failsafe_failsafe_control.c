#include "twinr_on_device_failsafe_internal.h"

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

static void recordLateralCommand(
    const float vx,
    const float vy,
    const twinrFsLateralCommandSource_t source)
{
  twinrFsLastCommandedVx = vx;
  twinrFsLastCommandedVy = vy;
  twinrFsLateralCommandSource = (uint8_t)source;
}

void sendFailsafeSetpoint(
    const float vx,
    const float vy,
    const float z,
    const twinrFsLateralCommandSource_t source)
{
  static setpoint_t setpoint;
  setHoverSetpoint(&setpoint, vx, vy, z);
  recordLateralCommand(vx, vy, source);
  commanderSetSetpoint(&setpoint, TWINR_FS_COMMANDER_PRIORITY);
}

void sendManualTakeoffSetpoint(
    const float thrust,
    const twinrFsLateralCommandSource_t source)
{
  static setpoint_t setpoint;
  memset(&setpoint, 0, sizeof(setpoint_t));
  setpoint.mode.x = modeDisable;
  setpoint.mode.y = modeDisable;
  setpoint.mode.z = modeDisable;
  setpoint.mode.roll = modeAbs;
  setpoint.mode.pitch = modeAbs;
  setpoint.mode.yaw = modeVelocity;
  setpoint.attitude.roll = 0.0f;
  setpoint.attitude.pitch = 0.0f;
  setpoint.attitudeRate.yaw = 0.0f;
  setpoint.thrust = clampFloat(thrust, 0.0f, (float)UINT16_MAX);
  recordLateralCommand(0.0f, 0.0f, source);
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

uint16_t minClearanceMm(const uint16_t frontMm,
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
                              const bool lateralClearanceArmed,
                              twinrFsReason_t *reason)
{
  if (!twinrFsRequireClearance) {
    return false;
  }
  if (rangeIsValid(upMm) && upMm < twinrFsMinUpClearanceMm) {
    *reason = twinrFsReasonUpClearance;
    return true;
  }
  if (!lateralClearanceArmed) {
    return false;
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

bool batteryViolated(const uint16_t vbatMv,
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

bool clearanceDebounced(const uint16_t frontMm,
                        const uint16_t backMm,
                        const uint16_t leftMm,
                        const uint16_t rightMm,
                        const uint16_t upMm,
                        const bool lateralClearanceArmed,
                        twinrFsReason_t *reason)
{
  twinrFsReason_t currentReason = twinrFsReasonNone;
  if (clearanceViolated(frontMm, backMm, leftMm, rightMm, upMm, lateralClearanceArmed, &currentReason)) {
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

bool heartbeatExpired(const TickType_t now)
{
  if (!twinrFsEnable || twinrFsLastHeartbeatTick == 0 || twinrFsHeartbeatTimeoutMs == 0U) {
    return false;
  }
  return (now - twinrFsLastHeartbeatTick) > M2T(twinrFsHeartbeatTimeoutMs);
}

void triggerFailsafe(const twinrFsReason_t reason,
                     const TickType_t now,
                     const float stateEstimateZ,
                     const uint16_t downMm)
{
  if (twinrFsControlActive) {
    return;
  }
  const bool rangeLive = missionRangeLive();
  const bool flowLive = missionFlowLive();

  float targetZ = stateEstimateZ;
  if (targetZ < TWINR_FS_MIN_ACTIVE_ALTITUDE_M && rangeIsValid(downMm)) {
    targetZ = ((float)downMm) / 1000.0f;
  }
  if (targetZ < TWINR_FS_MIN_ACTIVE_ALTITUDE_M) {
    targetZ = TWINR_FS_MIN_ACTIVE_ALTITUDE_M;
  }

  twinrFsFailsafeTargetZ = clampFloat(targetZ, TWINR_FS_MIN_ACTIVE_ALTITUDE_M, TWINR_FS_MAX_TARGET_Z_M);
  twinrFsTouchdownConfirmCount = 0U;
  resetMissionState();
  twinrFsControlActive = 1U;
  recordTriggerEvent(reason);
  setState(twinrFsStateFailsafeBrake,
           reason,
           now,
           true,
           flightObserved(true, downMm, stateEstimateZ),
           rangeLive,
           flowLive,
           stateEstimateZ);
  DEBUG_PRINT("Failsafe triggered reason=%u targetZ=%.3f\n",
              (unsigned)reason,
              (double)twinrFsFailsafeTargetZ);
}

void runFailsafeControl(const TickType_t now,
                        const float stateEstimateZ,
                        const uint16_t frontMm,
                        const uint16_t backMm,
                        const uint16_t leftMm,
                        const uint16_t rightMm,
                        const uint16_t downMm,
                        const uint16_t motionSqual,
                        const bool isFlying)
{
  (void)motionSqual;
  const bool airborne = flightObserved(isFlying, downMm, stateEstimateZ);
  const bool rangeLive = missionRangeLive();
  const bool flowLive = missionFlowLive();
  float vx = 0.0f;
  float vy = 0.0f;
  computeRepelVelocity(frontMm, backMm, leftMm, rightMm, &vx, &vy);

  if (twinrFsState == twinrFsStateFailsafeBrake) {
    sendFailsafeSetpoint(
        vx,
        vy,
        twinrFsFailsafeTargetZ,
        twinrFsLateralCommandSourceFailsafeBrake);
    if ((now - twinrFsLastStateTick) >= M2T(twinrFsBrakeHoldMs)) {
      setState(twinrFsStateFailsafeDescend, twinrFsReason, now, true, airborne, rangeLive, flowLive, stateEstimateZ);
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
    sendFailsafeSetpoint(
        vx,
        vy,
        twinrFsFailsafeTargetZ,
        twinrFsLateralCommandSourceFailsafeDescend);
    if ((rangeIsValid(downMm) && downMm <= TWINR_FS_TOUCHDOWN_HEIGHT_MM) || stateEstimateZ <= 0.03f) {
      setState(twinrFsStateTouchdownConfirm, twinrFsReason, now, true, airborne, rangeLive, flowLive, stateEstimateZ);
    }
    return;
  }

  if (twinrFsState == twinrFsStateTouchdownConfirm) {
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
    sendFailsafeSetpoint(
        0.0f,
        0.0f,
        0.0f,
        twinrFsLateralCommandSourceTouchdownConfirm);
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
        if (!twinrFsFlightObserved) {
          twinrFsGroundAbortQuiet = 1U;
        }
        recordLandedEvent(downMm);
        setState(twinrFsStateLanded, twinrFsReason, now, true, airborne, rangeLive, flowLive, stateEstimateZ);
      }
    }
  }
}
