#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "twinr_on_device_failsafe_disturbance_control.h"
#include "twinr_on_device_failsafe_vertical_control.h"

#define TWINR_FS_TAKEOFF_THRUST_ESTIMATE_DEFAULT 0.50f
#define TWINR_FS_TAKEOFF_THRUST_MIN 0.40f
#define TWINR_FS_TAKEOFF_THRUST_MAX 0.62f
#define TWINR_FS_TAKEOFF_HEIGHT_GAIN 1.10f
#define TWINR_FS_TAKEOFF_VZ_DAMPING 0.18f
#define TWINR_FS_TAKEOFF_PROGRESS_BOOST_MAX 0.10f
#define TWINR_FS_TAKEOFF_PROGRESS_RAMP_MS 300U
#define TWINR_FS_TAKEOFF_CEILING_WITHOUT_PROGRESS_MS 350U
#define TWINR_FS_HOVER_THRUST_EST_ALPHA 0.08f
#define TWINR_FS_HOVER_THRUST_EST_MAX_DELTA 0.08f
#define TWINR_FS_HOVER_THRUST_EST_VZ_QUIET_MPS 0.10f
#define TWINR_FS_HOVER_THRUST_EST_Z_ERR_M 0.03f
#define TWINR_FS_HOVER_MIN_RANGE_RISE_MM 20U
#define TWINR_FS_HOVER_BATTERY_FILTER_ALPHA 0.10f
#define TWINR_FS_DISTURBANCE_ALPHA 0.10f
#define TWINR_FS_DISTURBANCE_MAX_MPS 0.12f
#define TWINR_FS_DISTURBANCE_RECOVERABLE_MAX_MPS 0.10f
#define TWINR_FS_DISTURBANCE_ABORT_SAMPLES 5U

static twinrFsVerticalControlConfig_t verticalConfig(void)
{
  return (twinrFsVerticalControlConfig_t){
      .takeoffThrustEstimateDefault = TWINR_FS_TAKEOFF_THRUST_ESTIMATE_DEFAULT,
      .takeoffThrustMin = TWINR_FS_TAKEOFF_THRUST_MIN,
      .takeoffThrustMax = TWINR_FS_TAKEOFF_THRUST_MAX,
      .takeoffHeightGain = TWINR_FS_TAKEOFF_HEIGHT_GAIN,
      .takeoffVzDamping = TWINR_FS_TAKEOFF_VZ_DAMPING,
      .takeoffProgressBoostMax = TWINR_FS_TAKEOFF_PROGRESS_BOOST_MAX,
      .takeoffProgressRampMs = TWINR_FS_TAKEOFF_PROGRESS_RAMP_MS,
      .minRangeRiseMm = TWINR_FS_HOVER_MIN_RANGE_RISE_MM,
      .ceilingWithoutProgressMs = TWINR_FS_TAKEOFF_CEILING_WITHOUT_PROGRESS_MS,
      .batteryLowMv = 3550U,
      .batteryCriticalMv = 3350U,
      .hoverThrustEstAlpha = TWINR_FS_HOVER_THRUST_EST_ALPHA,
      .hoverThrustEstMaxDelta = TWINR_FS_HOVER_THRUST_EST_MAX_DELTA,
      .hoverThrustEstVzQuietMps = TWINR_FS_HOVER_THRUST_EST_VZ_QUIET_MPS,
      .hoverThrustEstZErrM = TWINR_FS_HOVER_THRUST_EST_Z_ERR_M,
      .batteryFilterAlpha = TWINR_FS_HOVER_BATTERY_FILTER_ALPHA,
  };
}

static twinrFsDisturbanceControlConfig_t disturbanceConfig(void)
{
  return (twinrFsDisturbanceControlConfig_t){
      .estimateAlpha = TWINR_FS_DISTURBANCE_ALPHA,
      .maxCommandMps = TWINR_FS_DISTURBANCE_MAX_MPS,
      .recoverableMagnitudeMps = TWINR_FS_DISTURBANCE_RECOVERABLE_MAX_MPS,
      .nonrecoverableSamples = TWINR_FS_DISTURBANCE_ABORT_SAMPLES,
  };
}

static int runVerticalHandoff(void)
{
  static const uint16_t downSamples[] = {24U, 24U, 27U, 31U, 38U, 47U, 57U, 68U, 79U};
  const twinrFsVerticalControlConfig_t config = verticalConfig();
  twinrFsVerticalControlState_t state = {0};
  twinrFsVerticalControlReset(&state, &config);

  twinrFsVerticalTakeoffDecision_t finalDecision = {0};
  size_t steps = 0U;
  for (size_t index = 0U; index < (sizeof(downSamples) / sizeof(downSamples[0])); index += 1U) {
    finalDecision = twinrFsVerticalControlStepTakeoff(
        &state,
        &config,
        &(const twinrFsVerticalTakeoffObservation_t){
            .vbatMv = 4040U,
            .downMm = downSamples[index],
            .baselineHeightMm = 24U,
            .microHeightMm = 80U,
            .targetToleranceMm = 50U,
            .takeoffRampMs = 300U,
            .loopPeriodMs = 20U,
            .phaseElapsedMs = (uint32_t)(index * 20U),
            .stateEstimateVz = 0.04f,
            .rangeLive = index >= 6U,
            .flowLive = index >= 6U,
            .attitudeQuiet = index >= 3U,
        });
    steps = index + 1U;
    if (finalDecision.shouldAbort || finalDecision.shouldHandoff) {
      break;
    }
  }

  printf(
      "{\"scenario\":\"vertical_handoff\",\"steps\":%zu,\"handoff\":%s,"
      "\"abort\":%s,\"progress_class\":%u,\"thrust_ratio\":%.5f,"
      "\"ceiling_ms\":%u,\"battery_limited\":%u}\n",
      steps,
      finalDecision.shouldHandoff ? "true" : "false",
      finalDecision.shouldAbort ? "true" : "false",
      (unsigned int)finalDecision.progressClass,
      finalDecision.thrustRatio,
      (unsigned int)state.ceilingWithoutProgressAccumMs,
      (unsigned int)state.batteryLimited);
  return 0;
}

static int runVerticalCeilingAbort(void)
{
  const twinrFsVerticalControlConfig_t config = verticalConfig();
  twinrFsVerticalControlState_t state = {0};
  twinrFsVerticalControlReset(&state, &config);

  twinrFsVerticalTakeoffDecision_t finalDecision = {0};
  size_t steps = 0U;
  for (size_t index = 0U; index < 40U; index += 1U) {
    finalDecision = twinrFsVerticalControlStepTakeoff(
        &state,
        &config,
        &(const twinrFsVerticalTakeoffObservation_t){
            .vbatMv = 3990U,
            .downMm = 24U,
            .baselineHeightMm = 24U,
            .microHeightMm = 80U,
            .targetToleranceMm = 50U,
            .takeoffRampMs = 300U,
            .loopPeriodMs = 20U,
            .phaseElapsedMs = (uint32_t)(index * 20U),
            .stateEstimateVz = 0.0f,
            .rangeLive = false,
            .flowLive = false,
            .attitudeQuiet = true,
        });
    steps = index + 1U;
    if (finalDecision.shouldAbort) {
      break;
    }
  }

  printf(
      "{\"scenario\":\"vertical_ceiling_abort\",\"steps\":%zu,\"handoff\":%s,"
      "\"abort\":%s,\"progress_class\":%u,\"thrust_ratio\":%.5f,"
      "\"ceiling_ms\":%u,\"thrust_at_ceiling\":%s}\n",
      steps,
      finalDecision.shouldHandoff ? "true" : "false",
      finalDecision.shouldAbort ? "true" : "false",
      (unsigned int)finalDecision.progressClass,
      finalDecision.thrustRatio,
      (unsigned int)state.ceilingWithoutProgressAccumMs,
      finalDecision.thrustAtCeiling ? "true" : "false");
  return 0;
}

static int runDisturbanceRecoverable(void)
{
  const twinrFsDisturbanceControlConfig_t config = disturbanceConfig();
  twinrFsDisturbanceControlState_t state = {0};
  twinrFsDisturbanceControlReset(&state);

  twinrFsDisturbanceDecision_t decision = {0};
  for (size_t index = 0U; index < 8U; index += 1U) {
    decision = twinrFsDisturbanceControlStep(
        &state,
        &config,
        &(const twinrFsDisturbanceObservation_t){
            .flowLive = true,
            .downMm = 150U,
            .stateEstimateVx = 0.05f,
            .stateEstimateVy = -0.03f,
        });
  }

  printf(
      "{\"scenario\":\"disturbance_recoverable\",\"valid\":%s,"
      "\"recoverable\":%s,\"abort\":%s,\"severity_class\":%u,"
      "\"severity_permille\":%u,\"near_ground\":%u,\"vx_command\":%.5f,"
      "\"vy_command\":%.5f}\n",
      decision.valid ? "true" : "false",
      decision.recoverable ? "true" : "false",
      decision.shouldAbort ? "true" : "false",
      (unsigned int)decision.severityClass,
      (unsigned int)decision.severityPermille,
      (unsigned int)decision.nearGroundObserved,
      decision.vxCommand,
      decision.vyCommand);
  return 0;
}

static int runDisturbanceNonrecoverable(void)
{
  const twinrFsDisturbanceControlConfig_t config = disturbanceConfig();
  twinrFsDisturbanceControlState_t state = {0};
  twinrFsDisturbanceControlReset(&state);

  twinrFsDisturbanceDecision_t decision = {0};
  size_t steps = 0U;
  for (size_t index = 0U; index < 32U; index += 1U) {
    decision = twinrFsDisturbanceControlStep(
        &state,
        &config,
        &(const twinrFsDisturbanceObservation_t){
            .flowLive = true,
            .downMm = 140U,
            .stateEstimateVx = 0.24f,
            .stateEstimateVy = 0.18f,
        });
    steps = index + 1U;
    if (decision.shouldAbort) {
      break;
    }
  }

  printf(
      "{\"scenario\":\"disturbance_nonrecoverable\",\"steps\":%zu,\"valid\":%s,"
      "\"recoverable\":%s,\"abort\":%s,\"severity_class\":%u,"
      "\"severity_permille\":%u,\"nonrecoverable_count\":%u}\n",
      steps,
      decision.valid ? "true" : "false",
      decision.recoverable ? "true" : "false",
      decision.shouldAbort ? "true" : "false",
      (unsigned int)decision.severityClass,
      (unsigned int)decision.severityPermille,
      (unsigned int)state.nonrecoverableCount);
  return 0;
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s <scenario>\n", argv[0]);
    return 2;
  }

  if (strcmp(argv[1], "vertical_handoff") == 0) {
    return runVerticalHandoff();
  }
  if (strcmp(argv[1], "vertical_ceiling_abort") == 0) {
    return runVerticalCeilingAbort();
  }
  if (strcmp(argv[1], "disturbance_recoverable") == 0) {
    return runDisturbanceRecoverable();
  }
  if (strcmp(argv[1], "disturbance_nonrecoverable") == 0) {
    return runDisturbanceNonrecoverable();
  }

  fprintf(stderr, "unknown scenario: %s\n", argv[1]);
  return 2;
}
