#include "twinr_on_device_failsafe_vertical_control.h"

#include <math.h>

#define TWINR_FS_VERTICAL_BATTERY_FULL_MV 4200.0f
#define TWINR_FS_VERTICAL_OBSERVED_THRUST_MAX 65535.0f

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

static uint16_t clampUint16(const uint32_t value)
{
  if (value >= UINT16_MAX) {
    return UINT16_MAX;
  }
  return (uint16_t)value;
}

static bool rangeIsValid(const uint16_t rangeMm)
{
  return rangeMm > 0U && rangeMm < 32000U;
}

static void updateFilteredBattery(twinrFsVerticalControlState_t *state,
                                  const twinrFsVerticalControlConfig_t *config,
                                  const uint16_t vbatMv)
{
  if (vbatMv == 0U) {
    return;
  }
  const float boundedAlpha = clampFloat(config->batteryFilterAlpha, 0.0f, 1.0f);
  if (!(state->filteredBatteryMv > 0.0f)) {
    state->filteredBatteryMv = (float)vbatMv;
  } else {
    state->filteredBatteryMv =
        state->filteredBatteryMv + boundedAlpha * (((float)vbatMv) - state->filteredBatteryMv);
  }
  state->batteryLimited = (uint8_t)(state->filteredBatteryMv <= (float)config->batteryLowMv ? 1U : 0U);
}

void twinrFsVerticalControlReset(twinrFsVerticalControlState_t *state,
                                 const twinrFsVerticalControlConfig_t *config)
{
  state->hoverThrustEstimate = clampFloat(
      config->takeoffThrustEstimateDefault,
      config->takeoffThrustMin,
      config->takeoffThrustMax);
  state->filteredBatteryMv = TWINR_FS_VERTICAL_BATTERY_FULL_MV;
  state->ceilingWithoutProgressAccumMs = 0U;
  state->progressClass = twinrFsVerticalProgressIdle;
  state->batteryLimited = 0U;
}

twinrFsVerticalTakeoffDecision_t twinrFsVerticalControlStepTakeoff(
    twinrFsVerticalControlState_t *state,
    const twinrFsVerticalControlConfig_t *config,
    const twinrFsVerticalTakeoffObservation_t *observation)
{
  updateFilteredBattery(state, config, observation->vbatMv);

  const uint16_t targetDeltaMm = observation->microHeightMm > observation->baselineHeightMm
                                     ? (uint16_t)(observation->microHeightMm - observation->baselineHeightMm)
                                     : 0U;
  uint16_t desiredRiseMm = targetDeltaMm;
  if (observation->takeoffRampMs > 0U && observation->phaseElapsedMs < observation->takeoffRampMs) {
    desiredRiseMm = clampUint16(
        ((uint32_t)targetDeltaMm * observation->phaseElapsedMs) / observation->takeoffRampMs);
  }

  uint16_t observedRiseMm = 0U;
  if (rangeIsValid(observation->downMm) && observation->downMm > observation->baselineHeightMm) {
    observedRiseMm = (uint16_t)(observation->downMm - observation->baselineHeightMm);
  }

  const float riseErrorM =
      ((float)((int32_t)desiredRiseMm - (int32_t)observedRiseMm)) / 1000.0f;
  float progressBoost = 0.0f;
  if (desiredRiseMm > observedRiseMm && config->takeoffProgressRampMs > 0U) {
    const uint32_t boundedElapsed = observation->phaseElapsedMs < config->takeoffProgressRampMs
                                        ? observation->phaseElapsedMs
                                        : config->takeoffProgressRampMs;
    progressBoost = config->takeoffProgressBoostMax
                    * (((float)boundedElapsed) / ((float)config->takeoffProgressRampMs));
  }

  float thrustRatio = clampFloat(
      state->hoverThrustEstimate,
      config->takeoffThrustMin,
      config->takeoffThrustMax);
  thrustRatio += config->takeoffHeightGain * riseErrorM;
  thrustRatio -= config->takeoffVzDamping * observation->stateEstimateVz;
  thrustRatio += progressBoost;

  bool overshoot = false;
  if (rangeIsValid(observation->downMm)) {
    const uint16_t overshootLimitMm =
        (uint16_t)(observation->microHeightMm + observation->targetToleranceMm);
    if (observation->downMm > overshootLimitMm) {
      const float overshootM =
          ((float)(observation->downMm - overshootLimitMm)) / 1000.0f;
      thrustRatio -= config->takeoffHeightGain * overshootM;
      overshoot = observation->downMm >
                  (uint16_t)(observation->microHeightMm + (2U * observation->targetToleranceMm));
    }
  }

  thrustRatio = clampFloat(thrustRatio, config->takeoffThrustMin, config->takeoffThrustMax);
  const bool thrustAtCeiling = thrustRatio >= (config->takeoffThrustMax - 1e-4f);
  const bool progressReady = observedRiseMm >= config->minRangeRiseMm;
  const bool shouldHandoff = observation->rangeLive && observation->flowLive && observation->attitudeQuiet;

  state->ceilingWithoutProgressAccumMs = (!progressReady && thrustAtCeiling)
                                             ? clampUint16((uint32_t)state->ceilingWithoutProgressAccumMs
                                                           + observation->loopPeriodMs)
                                             : 0U;

  bool shouldAbort = false;
  uint8_t progressClass = twinrFsVerticalProgressSeekingLift;
  if (shouldHandoff) {
    progressClass = twinrFsVerticalProgressQualified;
  } else if (overshoot) {
    progressClass = twinrFsVerticalProgressOvershoot;
    shouldAbort = true;
  } else if (state->ceilingWithoutProgressAccumMs >= config->ceilingWithoutProgressMs) {
    progressClass = twinrFsVerticalProgressCeilingWithoutProgress;
    shouldAbort = true;
  } else if (progressReady) {
    progressClass = twinrFsVerticalProgressRangeRising;
  }

  state->progressClass = progressClass;

  return (twinrFsVerticalTakeoffDecision_t){
      .thrustRatio = thrustRatio,
      .commandedHeightMm = observation->microHeightMm,
      .progressClass = progressClass,
      .batteryLimited = state->batteryLimited,
      .thrustAtCeiling = thrustAtCeiling,
      .shouldHandoff = shouldHandoff,
      .shouldAbort = shouldAbort,
  };
}

void twinrFsVerticalControlObserveHover(twinrFsVerticalControlState_t *state,
                                        const twinrFsVerticalControlConfig_t *config,
                                        const twinrFsVerticalHoverObservation_t *observation,
                                        const float targetHeightM)
{
  updateFilteredBattery(state, config, observation->vbatMv);

  const float observedThrustRatio = clampFloat(
      observation->observedThrust / TWINR_FS_VERTICAL_OBSERVED_THRUST_MAX,
      0.0f,
      1.0f);
  if (!(observedThrustRatio > 0.0f)) {
    return;
  }

  const float verticalError = fabsf(targetHeightM - observation->stateEstimateZ);
  const float verticalSpeed = fabsf(observation->stateEstimateVz);
  if (verticalError > config->hoverThrustEstZErrM) {
    return;
  }
  if (verticalSpeed > config->hoverThrustEstVzQuietMps) {
    return;
  }

  float delta = observedThrustRatio - state->hoverThrustEstimate;
  delta = clampFloat(delta, -config->hoverThrustEstMaxDelta, config->hoverThrustEstMaxDelta);
  state->hoverThrustEstimate = clampFloat(
      state->hoverThrustEstimate + config->hoverThrustEstAlpha * delta,
      config->takeoffThrustMin,
      config->takeoffThrustMax);
}
