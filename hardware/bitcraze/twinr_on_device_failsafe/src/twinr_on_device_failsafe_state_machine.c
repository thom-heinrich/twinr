#include "twinr_on_device_failsafe_internal.h"

static void syncObservation(const twinrFsObservation_t *observation, const TickType_t now)
{
  twinrFsLastVbatMv = observation->vbatMv;
  twinrFsLastPmState = observation->pmState;
  twinrFsLastFrontMm = observation->frontMm;
  twinrFsLastBackMm = observation->backMm;
  twinrFsLastLeftMm = observation->leftMm;
  twinrFsLastRightMm = observation->rightMm;
  twinrFsLastUpRangeMm = observation->upMm;
  twinrFsLastMinClearanceMm = minClearanceMm(observation->frontMm,
                                             observation->backMm,
                                             observation->leftMm,
                                             observation->rightMm,
                                             observation->upMm);
  twinrFsLastDownRangeMm = observation->downMm;
  twinrFsLastMotionSqual = observation->motionSqual;
  twinrFsHeartbeatAgeMs = twinrFsLastHeartbeatTick == 0
                            ? 0U
                            : clampUint16(T2M(now - twinrFsLastHeartbeatTick));
}

void runTwinrFsStateMachine(const TickType_t now, const twinrFsObservation_t *observation)
{
  twinrFsReason_t triggerReason = twinrFsReasonNone;

  syncObservation(observation, now);
  const bool airborne = flightObserved(observation->isFlying,
                                       observation->downMm,
                                       observation->stateEstimateZ);

  if (airborne) {
    twinrFsFlightObserved = 1U;
    twinrFsGroundAbortQuiet = 0U;
  }

  processAppchannelPackets(now, airborne, observation->downMm, observation->stateEstimateZ);
  updateMissionTruthCounters(observation->downMm, observation->motionSqual);

  const bool rangeLive = missionRangeLive();
  const bool flowLive = missionFlowLive();

  if (twinrFsControlActive) {
    runFailsafeControl(now,
                       observation->stateEstimateZ,
                       observation->frontMm,
                       observation->backMm,
                       observation->leftMm,
                       observation->rightMm,
                       observation->downMm,
                       observation->motionSqual,
                       observation->isFlying);
    maybeSendStatus(now, false, airborne, rangeLive, flowLive, observation->stateEstimateZ);
    return;
  }

  if (!twinrFsEnable) {
    twinrFsFlightObserved = 0U;
    twinrFsGroundAbortQuiet = 0U;
    resetTriggerCounters();
    resetMissionState();
    setState(twinrFsStateDisabled,
             twinrFsReasonManualDisable,
             now,
             false,
             airborne,
             rangeLive,
             flowLive,
             observation->stateEstimateZ);
    maybeSendStatus(now, false, airborne, rangeLive, flowLive, observation->stateEstimateZ);
    return;
  }

  if (twinrFsGroundAbortQuiet) {
    resetTriggerCounters();
    setState(twinrFsStateLanded,
             twinrFsReason,
             now,
             false,
             airborne,
             rangeLive,
             flowLive,
             observation->stateEstimateZ);
    maybeSendStatus(now, false, airborne, rangeLive, flowLive, observation->stateEstimateZ);
    return;
  }

  if (!airborne) {
    resetBatteryCounters();
    if (twinrFsMissionActive) {
      runMissionControl(now,
                        observation->stateEstimateZ,
                        observation->downMm,
                        observation->motionSqual,
                        observation->isFlying);
      maybeSendStatus(now, false, airborne, rangeLive, flowLive, observation->stateEstimateZ);
      return;
    }
    if (clearanceDebounced(observation->frontMm,
                           observation->backMm,
                           observation->leftMm,
                           observation->rightMm,
                           observation->upMm,
                           false,
                           &triggerReason)) {
      triggerFailsafe(triggerReason, now, observation->stateEstimateZ, observation->downMm);
      runFailsafeControl(now,
                         observation->stateEstimateZ,
                         observation->frontMm,
                         observation->backMm,
                         observation->leftMm,
                         observation->rightMm,
                         observation->downMm,
                         observation->motionSqual,
                         observation->isFlying);
      return;
    }
    if (twinrFsState != twinrFsStateMissionComplete) {
      setState(twinrFsStateMonitoring,
               twinrFsReasonNone,
               now,
               false,
               airborne,
               rangeLive,
               flowLive,
               observation->stateEstimateZ);
    }
    maybeSendStatus(now, false, airborne, rangeLive, flowLive, observation->stateEstimateZ);
    return;
  }

  if (batteryViolated(observation->vbatMv, observation->pmState, &triggerReason)) {
    /* triggerReason already set */
  } else if (clearanceDebounced(observation->frontMm,
                                observation->backMm,
                                observation->leftMm,
                                observation->rightMm,
                                observation->upMm,
                                (twinrFsArmLateralClearance != 0U) || twinrFsMissionHoverQualified,
                                &triggerReason)) {
    /* triggerReason already set */
  } else if (heartbeatExpired(now)) {
    triggerReason = twinrFsReasonHeartbeatLoss;
  }

  if (triggerReason != twinrFsReasonNone) {
    triggerFailsafe(triggerReason, now, observation->stateEstimateZ, observation->downMm);
    runFailsafeControl(now,
                       observation->stateEstimateZ,
                       observation->frontMm,
                       observation->backMm,
                       observation->leftMm,
                       observation->rightMm,
                       observation->downMm,
                       observation->motionSqual,
                       observation->isFlying);
    return;
  }

  if (twinrFsMissionActive) {
    runMissionControl(now,
                      observation->stateEstimateZ,
                      observation->downMm,
                      observation->motionSqual,
                      observation->isFlying);
    return;
  }

  setState(twinrFsStateMonitoring,
           twinrFsReasonNone,
           now,
           false,
           airborne,
           rangeLive,
           flowLive,
           observation->stateEstimateZ);
  maybeSendStatus(now, false, airborne, rangeLive, flowLive, observation->stateEstimateZ);
}
