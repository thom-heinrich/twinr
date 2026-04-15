#ifndef TWINR_ON_DEVICE_FAILSAFE_DISTURBANCE_CONTROL_H
#define TWINR_ON_DEVICE_FAILSAFE_DISTURBANCE_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

/**
 * One bounded, firmware-agnostic lateral disturbance observer.
 *
 * The STM32 mission lane owns phase transitions and failure policy. This module
 * only estimates persistent lateral bias, classifies its severity, and
 * proposes one bounded compensation command.
 */

typedef enum {
  twinrFsDisturbanceSeverityNone = 0,
  twinrFsDisturbanceSeverityRecoverable = 1,
  twinrFsDisturbanceSeverityNonrecoverable = 2,
} twinrFsDisturbanceSeverityClass_t;

typedef struct {
  float estimateAlpha;
  float maxCommandMps;
  float recoverableMagnitudeMps;
  uint8_t nonrecoverableSamples;
} twinrFsDisturbanceControlConfig_t;

typedef struct {
  float estimateVx;
  float estimateVy;
  uint16_t severityPermille;
  uint8_t nonrecoverableCount;
  uint8_t recoverable;
  uint8_t severityClass;
  uint8_t nearGroundObserved;
} twinrFsDisturbanceControlState_t;

typedef struct {
  bool flowLive;
  uint16_t downMm;
  float stateEstimateVx;
  float stateEstimateVy;
} twinrFsDisturbanceObservation_t;

typedef struct {
  float vxCommand;
  float vyCommand;
  uint16_t severityPermille;
  uint8_t recoverable;
  uint8_t severityClass;
  uint8_t nearGroundObserved;
  bool valid;
  bool shouldAbort;
} twinrFsDisturbanceDecision_t;

void twinrFsDisturbanceControlReset(twinrFsDisturbanceControlState_t *state);

twinrFsDisturbanceDecision_t twinrFsDisturbanceControlStep(
    twinrFsDisturbanceControlState_t *state,
    const twinrFsDisturbanceControlConfig_t *config,
    const twinrFsDisturbanceObservation_t *observation);

#endif
