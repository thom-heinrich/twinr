#ifndef TWINR_ON_DEVICE_FAILSAFE_VERTICAL_CONTROL_H
#define TWINR_ON_DEVICE_FAILSAFE_VERTICAL_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

/**
 * One bounded, firmware-agnostic vertical bootstrap/hover-thrust model.
 *
 * This module stays pure C on purpose: it owns the vertical estimation and
 * decision logic without depending on FreeRTOS, Crazyflie app headers, or the
 * surrounding mission state machine. The STM32 mission lane feeds
 * observations in, consumes explicit decisions, and remains the single owner of
 * state transitions.
 */

typedef enum {
  twinrFsVerticalProgressIdle = 0,
  twinrFsVerticalProgressSeekingLift = 1,
  twinrFsVerticalProgressRangeRising = 2,
  twinrFsVerticalProgressQualified = 3,
  twinrFsVerticalProgressCeilingWithoutProgress = 4,
  twinrFsVerticalProgressOvershoot = 5,
} twinrFsVerticalProgressClass_t;

typedef struct {
  float takeoffThrustEstimateDefault;
  float takeoffThrustMin;
  float takeoffThrustMax;
  float takeoffHeightGain;
  float takeoffVzDamping;
  float takeoffProgressBoostMax;
  uint16_t takeoffProgressRampMs;
  uint16_t minRangeRiseMm;
  uint16_t ceilingWithoutProgressMs;
  uint16_t batteryLowMv;
  uint16_t batteryCriticalMv;
  float hoverThrustEstAlpha;
  float hoverThrustEstMaxDelta;
  float hoverThrustEstVzQuietMps;
  float hoverThrustEstZErrM;
  float batteryFilterAlpha;
} twinrFsVerticalControlConfig_t;

typedef struct {
  float hoverThrustEstimate;
  float filteredBatteryMv;
  uint16_t ceilingWithoutProgressAccumMs;
  uint8_t progressClass;
  uint8_t batteryLimited;
} twinrFsVerticalControlState_t;

typedef struct {
  uint16_t vbatMv;
  uint16_t downMm;
  uint16_t baselineHeightMm;
  uint16_t microHeightMm;
  uint16_t targetToleranceMm;
  uint16_t takeoffRampMs;
  uint16_t loopPeriodMs;
  uint32_t phaseElapsedMs;
  float stateEstimateVz;
  bool rangeLive;
  bool flowLive;
  bool attitudeQuiet;
} twinrFsVerticalTakeoffObservation_t;

typedef struct {
  float stateEstimateZ;
  float stateEstimateVz;
  float observedThrust;
  uint16_t vbatMv;
} twinrFsVerticalHoverObservation_t;

typedef struct {
  float thrustRatio;
  uint16_t commandedHeightMm;
  uint8_t progressClass;
  uint8_t batteryLimited;
  bool thrustAtCeiling;
  bool shouldHandoff;
  bool shouldAbort;
} twinrFsVerticalTakeoffDecision_t;

void twinrFsVerticalControlReset(twinrFsVerticalControlState_t *state,
                                 const twinrFsVerticalControlConfig_t *config);

twinrFsVerticalTakeoffDecision_t twinrFsVerticalControlStepTakeoff(
    twinrFsVerticalControlState_t *state,
    const twinrFsVerticalControlConfig_t *config,
    const twinrFsVerticalTakeoffObservation_t *observation);

void twinrFsVerticalControlObserveHover(twinrFsVerticalControlState_t *state,
                                        const twinrFsVerticalControlConfig_t *config,
                                        const twinrFsVerticalHoverObservation_t *observation,
                                        float targetHeightM);

#endif
