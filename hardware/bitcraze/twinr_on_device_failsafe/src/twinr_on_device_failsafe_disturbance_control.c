#include "twinr_on_device_failsafe_disturbance_control.h"

#include <math.h>

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

void twinrFsDisturbanceControlReset(twinrFsDisturbanceControlState_t *state)
{
  state->estimateVx = 0.0f;
  state->estimateVy = 0.0f;
  state->severityPermille = 0U;
  state->nonrecoverableCount = 0U;
  state->recoverable = 0U;
  state->severityClass = twinrFsDisturbanceSeverityNone;
  state->nearGroundObserved = 0U;
}

twinrFsDisturbanceDecision_t twinrFsDisturbanceControlStep(
    twinrFsDisturbanceControlState_t *state,
    const twinrFsDisturbanceControlConfig_t *config,
    const twinrFsDisturbanceObservation_t *observation)
{
  if (!observation->flowLive) {
    twinrFsDisturbanceControlReset(state);
    return (twinrFsDisturbanceDecision_t){
        .valid = false,
    };
  }

  state->estimateVx = state->estimateVx
                      + config->estimateAlpha * (observation->stateEstimateVx - state->estimateVx);
  state->estimateVy = state->estimateVy
                      + config->estimateAlpha * (observation->stateEstimateVy - state->estimateVy);

  const float magnitude = hypotf(state->estimateVx, state->estimateVy);
  const float boundedMagnitude = clampFloat(
      magnitude,
      0.0f,
      config->recoverableMagnitudeMps);
  state->severityPermille = (uint16_t)lrintf(
      (boundedMagnitude / config->recoverableMagnitudeMps) * 1000.0f);
  state->recoverable = (uint8_t)(magnitude <= config->recoverableMagnitudeMps ? 1U : 0U);
  state->severityClass = (uint8_t)(magnitude <= 1e-4f
                                       ? twinrFsDisturbanceSeverityNone
                                       : (state->recoverable
                                              ? twinrFsDisturbanceSeverityRecoverable
                                              : twinrFsDisturbanceSeverityNonrecoverable));
  state->nearGroundObserved = (uint8_t)(observation->downMm > 0U && observation->downMm < 80U ? 1U : 0U);

  if (!state->recoverable && state->nonrecoverableCount < UINT8_MAX) {
    state->nonrecoverableCount += 1U;
  } else if (state->recoverable) {
    state->nonrecoverableCount = 0U;
  }

  return (twinrFsDisturbanceDecision_t){
      .vxCommand = clampFloat(-state->estimateVx, -config->maxCommandMps, config->maxCommandMps),
      .vyCommand = clampFloat(-state->estimateVy, -config->maxCommandMps, config->maxCommandMps),
      .severityPermille = state->severityPermille,
      .recoverable = state->recoverable,
      .severityClass = state->severityClass,
      .nearGroundObserved = state->nearGroundObserved,
      .valid = true,
      .shouldAbort = state->nonrecoverableCount >= config->nonrecoverableSamples,
  };
}
