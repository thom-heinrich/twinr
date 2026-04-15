#ifndef TWINR_ON_DEVICE_FAILSAFE_INTERNAL_H
#define TWINR_ON_DEVICE_FAILSAFE_INTERNAL_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "FreeRTOS.h"
#include "app.h"
#include "app_channel.h"
#include "commander.h"
#include "debug.h"
#include "eventtrigger.h"
#include "log.h"
#include "param.h"
#include "supervisor.h"
#include "task.h"
#include "twinr_on_device_failsafe_disturbance_control.h"
#include "twinr_on_device_failsafe_vertical_control.h"

#define DEBUG_MODULE "TWINRFS"

#define TWINR_FS_PROTOCOL_VERSION 4U
#define TWINR_FS_PROTOCOL_VERSION_LEGACY 1U
#define TWINR_FS_PROTOCOL_VERSION_HEARTBEAT_V2 2U
#define TWINR_FS_PACKET_KIND_HEARTBEAT 1U
#define TWINR_FS_PACKET_KIND_STATUS 2U
#define TWINR_FS_PACKET_KIND_COMMAND 3U
#define TWINR_FS_FLAG_ENABLE (1U << 0)
#define TWINR_FS_FLAG_REQUIRE_CLEARANCE (1U << 1)
#define TWINR_FS_FLAG_ARM_LATERAL_CLEARANCE (1U << 2)

#define TWINR_FS_LOOP_PERIOD_MS 20U
#define TWINR_FS_STATUS_PERIOD_MS 100U
#define TWINR_FS_HEARTBEAT_TIMEOUT_MS 350U
#define TWINR_FS_HEARTBEAT_TIMEOUT_MIN_MS 100U
#define TWINR_FS_HEARTBEAT_TIMEOUT_MAX_MS 5000U
#define TWINR_FS_LOW_BATTERY_MV 3550U
#define TWINR_FS_CRITICAL_BATTERY_MV 3350U
#define TWINR_FS_BATTERY_THRESHOLD_MIN_MV 3000U
#define TWINR_FS_BATTERY_THRESHOLD_MAX_MV 4300U
#define TWINR_FS_BATTERY_HYSTERESIS_MV 60U
#define TWINR_FS_MIN_CLEARANCE_MM 350U
#define TWINR_FS_MIN_UP_CLEARANCE_MM 250U
#define TWINR_FS_CLEARANCE_MIN_MM 100U
#define TWINR_FS_CLEARANCE_MAX_MM 3000U
#define TWINR_FS_DESCENT_RATE_MMPS 120U
#define TWINR_FS_DESCENT_RATE_MIN_MMPS 40U
#define TWINR_FS_DESCENT_RATE_MAX_MMPS 500U
#define TWINR_FS_MAX_REPEL_VELOCITY_MMPS 150U
#define TWINR_FS_MAX_REPEL_VELOCITY_LIMIT_MMPS 500U
#define TWINR_FS_BRAKE_HOLD_MS 200U
#define TWINR_FS_BRAKE_HOLD_MAX_MS 2000U
#define TWINR_FS_RANGE_INVALID_MM 32000U
#define TWINR_FS_LANDING_FLOOR_MM 80U
#define TWINR_FS_TOUCHDOWN_HEIGHT_MM 30U
#define TWINR_FS_TOUCHDOWN_CONFIRM_MM 50U
#define TWINR_FS_TOUCHDOWN_CONFIRM_SAMPLES 3U
#define TWINR_FS_MIN_ACTIVE_ALTITUDE_M 0.08f
#define TWINR_FS_MAX_TARGET_Z_M 1.20f
#define TWINR_FS_COMMANDER_PRIORITY COMMANDER_PRIORITY_EXTRX
#define TWINR_FS_LOW_BATTERY_DEBOUNCE_TICKS_DEFAULT 25U
#define TWINR_FS_CRITICAL_BATTERY_DEBOUNCE_TICKS_DEFAULT 6U
#define TWINR_FS_CLEARANCE_DEBOUNCE_TICKS_DEFAULT 2U
#define TWINR_FS_DEBOUNCE_TICKS_MIN 1U
#define TWINR_FS_DEBOUNCE_TICKS_MAX 50U
#define TWINR_FS_HOVER_COMMAND_KIND_START 1U
#define TWINR_FS_HOVER_COMMAND_KIND_LAND 2U
#define TWINR_FS_HOVER_COMMAND_KIND_ABORT 3U
#define TWINR_FS_HOVER_STATE_FLAG_AIRBORNE (1U << 0)
#define TWINR_FS_HOVER_STATE_FLAG_RANGE_LIVE (1U << 1)
#define TWINR_FS_HOVER_STATE_FLAG_FLOW_LIVE (1U << 2)
#define TWINR_FS_HOVER_STATE_FLAG_MISSION_ACTIVE (1U << 3)
#define TWINR_FS_HOVER_STATE_FLAG_TAKEOFF_PROVEN (1U << 4)
#define TWINR_FS_HOVER_STATE_FLAG_HOVER_QUALIFIED (1U << 5)
#define TWINR_FS_HOVER_STATE_FLAG_LANDING_ACTIVE (1U << 6)
#define TWINR_FS_HOVER_STATE_FLAG_COMPLETE (1U << 7)
#define TWINR_FS_HOVER_TARGET_HEIGHT_MM_DEFAULT 100U
#define TWINR_FS_HOVER_MICRO_HEIGHT_MM_DEFAULT 80U
#define TWINR_FS_HOVER_TARGET_TOLERANCE_MM_DEFAULT 50U
#define TWINR_FS_HOVER_TAKEOFF_RAMP_MS_DEFAULT 1000U
#define TWINR_FS_HOVER_DURATION_MS_DEFAULT 1000U
#define TWINR_FS_HOVER_TAKEOFF_TIMEOUT_MS 2500U
#define TWINR_FS_HOVER_FLOW_GATE_MIN_HEIGHT_MM 80U
#define TWINR_FS_HOVER_MIN_RANGE_RISE_MM 20U
#define TWINR_FS_HOVER_RANGE_FRESH_SAMPLES 2U
#define TWINR_FS_HOVER_RANGE_RISE_SAMPLES 2U
#define TWINR_FS_HOVER_FLOW_LIVE_SAMPLES 2U
#define TWINR_FS_HOVER_ATTITUDE_QUIET_SAMPLES 3U
#define TWINR_FS_HOVER_TRUTH_STALE_SAMPLES TWINR_FS_HOVER_FLOW_LIVE_SAMPLES
#define TWINR_FS_HOVER_TRUTH_FLAP_LIMIT 3U
#define TWINR_FS_HOVER_FLOW_MIN_SQUAL 1U
#define TWINR_FS_HOVER_ATTITUDE_QUIET_MAX_DEG 5.0f

#define TWINR_FS_DEBUG_FLAG_RANGE_READY (1U << 0)
#define TWINR_FS_DEBUG_FLAG_FLOW_READY (1U << 1)
#define TWINR_FS_DEBUG_FLAG_THRUST_AT_CEILING (1U << 2)
#define TWINR_FS_DEBUG_FLAG_HOVER_THRUST_VALID (1U << 3)
#define TWINR_FS_DEBUG_FLAG_DISTURBANCE_VALID (1U << 4)
#define TWINR_FS_DEBUG_FLAG_TOUCHDOWN_BY_RANGE (1U << 5)
#define TWINR_FS_DEBUG_FLAG_TOUCHDOWN_BY_SUPERVISOR (1U << 6)
#define TWINR_FS_DEBUG_FLAG_ATTITUDE_READY (1U << 7)

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
#define TWINR_FS_HOVER_BATTERY_FILTER_ALPHA 0.10f
#define TWINR_FS_DISTURBANCE_ALPHA 0.10f
#define TWINR_FS_DISTURBANCE_MAX_MPS 0.12f
#define TWINR_FS_DISTURBANCE_RECOVERABLE_MAX_MPS 0.10f
#define TWINR_FS_DISTURBANCE_ABORT_SAMPLES 5U

typedef enum {
  twinrFsStateDisabled = 0,
  twinrFsStateMonitoring = 1,
  twinrFsStateFailsafeBrake = 2,
  twinrFsStateFailsafeDescend = 3,
  twinrFsStateTouchdownConfirm = 4,
  twinrFsStateLanded = 5,
  twinrFsStateMissionTakeoff = 6,
  twinrFsStateMissionHover = 7,
  twinrFsStateMissionLanding = 8,
  twinrFsStateMissionComplete = 9,
} twinrFsState_t;

typedef enum {
  twinrFsReasonNone = 0,
  twinrFsReasonHeartbeatLoss = 1,
  twinrFsReasonLowBattery = 2,
  twinrFsReasonCriticalBattery = 3,
  twinrFsReasonClearance = 4,
  twinrFsReasonUpClearance = 5,
  twinrFsReasonManualDisable = 6,
  twinrFsReasonMissionAbort = 7,
  twinrFsReasonTakeoffRangeLiveness = 8,
  twinrFsReasonTakeoffFlowLiveness = 9,
  twinrFsReasonTakeoffAttitudeQuiet = 10,
  twinrFsReasonTruthStale = 11,
  twinrFsReasonStateFlapping = 12,
  twinrFsReasonCeilingWithoutProgress = 13,
  twinrFsReasonDisturbanceNonrecoverable = 14,
  twinrFsReasonTakeoffOvershoot = 15,
} twinrFsReason_t;

typedef enum {
  twinrFsLateralCommandSourceNone = 0,
  twinrFsLateralCommandSourceMissionTakeoff = 1,
  twinrFsLateralCommandSourceMissionHover = 2,
  twinrFsLateralCommandSourceMissionLanding = 3,
  twinrFsLateralCommandSourceFailsafeBrake = 4,
  twinrFsLateralCommandSourceFailsafeDescend = 5,
  twinrFsLateralCommandSourceTouchdownConfirm = 6,
} twinrFsLateralCommandSource_t;

typedef enum {
  twinrFsRejectNone = 0,
  twinrFsRejectMalformed = 1,
  twinrFsRejectWrongKind = 2,
  twinrFsRejectUnsupportedVersion = 3,
  twinrFsRejectSessionMismatch = 4,
  twinrFsRejectStaleSequence = 5,
  twinrFsRejectInFlightReconfigure = 6,
  twinrFsRejectInFlightDisable = 7,
  twinrFsRejectDowngrade = 8,
  twinrFsRejectWrongSessionCommand = 9,
  twinrFsRejectWrongCommand = 10,
  twinrFsRejectMissionBusy = 11,
} twinrFsReject_t;

typedef struct {
  uint8_t version;
  uint8_t packetKind;
  uint8_t flags;
  uint8_t reserved;
  uint16_t sessionId;
  uint16_t heartbeatTimeoutMs;
  uint16_t lowBatteryMv;
  uint16_t criticalBatteryMv;
  uint16_t minClearanceMm;
  uint16_t minUpClearanceMm;
  uint16_t descentRateMmps;
  uint16_t maxRepelVelocityMmps;
  uint16_t brakeHoldMs;
} __attribute__((packed)) twinrFsHeartbeatPacketV1_t;

typedef struct {
  uint8_t version;
  uint8_t packetKind;
  uint8_t flags;
  uint8_t reserved;
  uint16_t sessionId;
  uint16_t heartbeatTimeoutMs;
  uint16_t lowBatteryMv;
  uint16_t criticalBatteryMv;
  uint16_t minClearanceMm;
  uint16_t minUpClearanceMm;
  uint16_t descentRateMmps;
  uint16_t maxRepelVelocityMmps;
  uint16_t brakeHoldMs;
  uint8_t lowBatteryDebounceTicks;
  uint8_t criticalBatteryDebounceTicks;
  uint8_t clearanceDebounceTicks;
  uint8_t reserved2;
  uint32_t sequence;
} __attribute__((packed)) twinrFsHeartbeatPacketV2_t;

typedef struct {
  uint8_t version;
  uint8_t flags;
  uint16_t sessionId;
  uint16_t heartbeatTimeoutMs;
  uint16_t lowBatteryMv;
  uint16_t criticalBatteryMv;
  uint16_t minClearanceMm;
  uint16_t minUpClearanceMm;
  uint16_t descentRateMmps;
  uint16_t maxRepelVelocityMmps;
  uint16_t brakeHoldMs;
  uint8_t lowBatteryDebounceTicks;
  uint8_t criticalBatteryDebounceTicks;
  uint8_t clearanceDebounceTicks;
  bool hasSequence;
  uint32_t sequence;
} twinrFsHeartbeatConfig_t;

typedef struct {
  uint8_t version;
  uint8_t packetKind;
  uint8_t commandKind;
  uint8_t reserved;
  uint16_t sessionId;
  uint16_t targetHeightMm;
  uint16_t hoverDurationMs;
  uint16_t takeoffRampMs;
  uint16_t microHeightMm;
  uint16_t targetToleranceMm;
} __attribute__((packed)) twinrFsCommandPacket_t;

typedef struct {
  uint8_t version;
  uint8_t packetKind;
  uint8_t state;
  uint8_t reason;
  uint16_t sessionId;
  uint16_t heartbeatAgeMs;
  uint16_t vbatMv;
  uint16_t minClearanceMm;
  uint16_t downRangeMm;
  uint8_t missionFlags;
  uint8_t debugFlags;
  uint16_t targetHeightMm;
  uint16_t commandedHeightMm;
  uint16_t stateEstimateZMm;
  uint16_t upRangeMm;
  uint16_t motionSqual;
  uint8_t touchdownConfirmCount;
  uint8_t reserved;
  uint16_t hoverThrustPermille;
} __attribute__((packed)) twinrFsStatusPacket_t;

typedef struct {
  uint8_t protocolVersion;
  uint8_t enable;
  uint8_t requireClearance;
  uint8_t armLateralClearance;
  uint16_t sessionId;
  uint8_t sessionBound;
  uint8_t packetVersionInUse;
  uint16_t heartbeatTimeoutMs;
  uint16_t lowBatteryMv;
  uint16_t criticalBatteryMv;
  uint16_t minClearanceMm;
  uint16_t minUpClearanceMm;
  uint16_t descentRateMmps;
  uint16_t maxRepelVelocityMmps;
  uint16_t brakeHoldMs;
  uint8_t lowBatteryDebounceTicks;
  uint8_t criticalBatteryDebounceTicks;
  uint8_t clearanceDebounceTicks;
  uint8_t state;
  uint8_t reason;
  uint16_t heartbeatAgeMs;
  uint16_t lastVbatMv;
  uint16_t lastMinClearanceMm;
  uint16_t lastDownRangeMm;
  uint16_t lastUpRangeMm;
  uint16_t lastFrontMm;
  uint16_t lastBackMm;
  uint16_t lastLeftMm;
  uint16_t lastRightMm;
  uint8_t lastPmState;
  uint8_t controlActive;
  uint8_t lastRejectCode;
  uint16_t rejectedPacketCount;
  uint16_t overflowCount;
  uint32_t lastHeartbeatSeq;
  uint8_t heartbeatSeqValid;
  uint8_t missionActive;
  uint8_t missionLandRequested;
  uint8_t missionTakeoffProven;
  uint8_t missionHoverQualified;
  uint16_t missionTargetHeightMm;
  uint16_t missionMicroHeightMm;
  uint16_t missionTargetToleranceMm;
  uint16_t missionHoverDurationMs;
  uint16_t missionTakeoffRampMs;
  uint16_t missionBaselineHeightMm;
  uint16_t missionCommandedHeightMm;
  uint16_t lastMotionSqual;
  TickType_t lastHeartbeatTick;
  TickType_t lastStateTick;
  TickType_t lastStatusTick;
  TickType_t missionStartTick;
  TickType_t missionPhaseStartTick;
  float failsafeTargetZ;
  uint8_t touchdownConfirmCount;
  uint8_t missionTouchdownConfirmCount;
  uint8_t missionRangeFreshCount;
  uint8_t missionRangeRiseCount;
  uint8_t missionFlowReadyCount;
  uint8_t missionAttitudeQuietCount;
  uint8_t missionTruthStaleCount;
  uint8_t missionTruthFlapCount;
  uint8_t missionRangeFreshRaw;
  uint8_t missionRangeRiseRaw;
  uint8_t missionFlowReadyRaw;
  uint8_t missionAttitudeQuietRaw;
  uint8_t lowBatteryCount;
  uint8_t criticalBatteryCount;
  uint8_t clearanceCount;
  uint8_t pendingClearanceReason;
  uint8_t flightObserved;
  uint8_t groundAbortQuiet;
  logVarId_t frontId;
  logVarId_t backId;
  logVarId_t leftId;
  logVarId_t rightId;
  logVarId_t upId;
  logVarId_t downId;
  logVarId_t vbatMvId;
  logVarId_t pmStateId;
  logVarId_t stateEstimateZId;
  logVarId_t stateEstimateVxId;
  logVarId_t stateEstimateVyId;
  logVarId_t stateEstimateVzId;
  logVarId_t motionSqualId;
  logVarId_t stabilizerRollId;
  logVarId_t stabilizerPitchId;
  logVarId_t stabilizerThrustId;
  float lastStateEstimateVx;
  float lastStateEstimateVy;
  float lastStateEstimateVz;
  float lastRollDeg;
  float lastPitchDeg;
  float lastCommandedVx;
  float lastCommandedVy;
  uint16_t lastObservedThrust;
  twinrFsVerticalControlState_t verticalControl;
  twinrFsDisturbanceControlState_t disturbanceControl;
  uint8_t takeoffDebugFlags;
  uint8_t lateralCommandSource;
} twinrFsContext_t;

typedef struct {
  uint16_t frontMm;
  uint16_t backMm;
  uint16_t leftMm;
  uint16_t rightMm;
  uint16_t upMm;
  uint16_t downMm;
  uint16_t vbatMv;
  uint16_t motionSqual;
  uint8_t pmState;
  bool isFlying;
  float stateEstimateZ;
} twinrFsObservation_t;

extern twinrFsContext_t twinrFs;

#define twinrFsProtocolVersion twinrFs.protocolVersion
#define twinrFsEnable twinrFs.enable
#define twinrFsRequireClearance twinrFs.requireClearance
#define twinrFsArmLateralClearance twinrFs.armLateralClearance
#define twinrFsSessionId twinrFs.sessionId
#define twinrFsSessionBound twinrFs.sessionBound
#define twinrFsPacketVersionInUse twinrFs.packetVersionInUse
#define twinrFsHeartbeatTimeoutMs twinrFs.heartbeatTimeoutMs
#define twinrFsLowBatteryMv twinrFs.lowBatteryMv
#define twinrFsCriticalBatteryMv twinrFs.criticalBatteryMv
#define twinrFsMinClearanceMm twinrFs.minClearanceMm
#define twinrFsMinUpClearanceMm twinrFs.minUpClearanceMm
#define twinrFsDescentRateMmps twinrFs.descentRateMmps
#define twinrFsMaxRepelVelocityMmps twinrFs.maxRepelVelocityMmps
#define twinrFsBrakeHoldMs twinrFs.brakeHoldMs
#define twinrFsLowBatteryDebounceTicks twinrFs.lowBatteryDebounceTicks
#define twinrFsCriticalBatteryDebounceTicks twinrFs.criticalBatteryDebounceTicks
#define twinrFsClearanceDebounceTicks twinrFs.clearanceDebounceTicks
#define twinrFsState twinrFs.state
#define twinrFsReason twinrFs.reason
#define twinrFsHeartbeatAgeMs twinrFs.heartbeatAgeMs
#define twinrFsLastVbatMv twinrFs.lastVbatMv
#define twinrFsLastMinClearanceMm twinrFs.lastMinClearanceMm
#define twinrFsLastDownRangeMm twinrFs.lastDownRangeMm
#define twinrFsLastUpRangeMm twinrFs.lastUpRangeMm
#define twinrFsLastFrontMm twinrFs.lastFrontMm
#define twinrFsLastBackMm twinrFs.lastBackMm
#define twinrFsLastLeftMm twinrFs.lastLeftMm
#define twinrFsLastRightMm twinrFs.lastRightMm
#define twinrFsLastPmState twinrFs.lastPmState
#define twinrFsControlActive twinrFs.controlActive
#define twinrFsLastRejectCode twinrFs.lastRejectCode
#define twinrFsRejectedPacketCount twinrFs.rejectedPacketCount
#define twinrFsOverflowCount twinrFs.overflowCount
#define twinrFsLastHeartbeatSeq twinrFs.lastHeartbeatSeq
#define twinrFsHeartbeatSeqValid twinrFs.heartbeatSeqValid
#define twinrFsMissionActive twinrFs.missionActive
#define twinrFsMissionLandRequested twinrFs.missionLandRequested
#define twinrFsMissionTakeoffProven twinrFs.missionTakeoffProven
#define twinrFsMissionHoverQualified twinrFs.missionHoverQualified
#define twinrFsMissionTargetHeightMm twinrFs.missionTargetHeightMm
#define twinrFsMissionMicroHeightMm twinrFs.missionMicroHeightMm
#define twinrFsMissionTargetToleranceMm twinrFs.missionTargetToleranceMm
#define twinrFsMissionHoverDurationMs twinrFs.missionHoverDurationMs
#define twinrFsMissionTakeoffRampMs twinrFs.missionTakeoffRampMs
#define twinrFsMissionBaselineHeightMm twinrFs.missionBaselineHeightMm
#define twinrFsMissionCommandedHeightMm twinrFs.missionCommandedHeightMm
#define twinrFsLastMotionSqual twinrFs.lastMotionSqual
#define twinrFsLastHeartbeatTick twinrFs.lastHeartbeatTick
#define twinrFsLastStateTick twinrFs.lastStateTick
#define twinrFsLastStatusTick twinrFs.lastStatusTick
#define twinrFsMissionStartTick twinrFs.missionStartTick
#define twinrFsMissionPhaseStartTick twinrFs.missionPhaseStartTick
#define twinrFsFailsafeTargetZ twinrFs.failsafeTargetZ
#define twinrFsTouchdownConfirmCount twinrFs.touchdownConfirmCount
#define twinrFsMissionTouchdownConfirmCount twinrFs.missionTouchdownConfirmCount
#define twinrFsMissionRangeFreshCount twinrFs.missionRangeFreshCount
#define twinrFsMissionRangeRiseCount twinrFs.missionRangeRiseCount
#define twinrFsMissionFlowReadyCount twinrFs.missionFlowReadyCount
#define twinrFsMissionAttitudeQuietCount twinrFs.missionAttitudeQuietCount
#define twinrFsMissionTruthStaleCount twinrFs.missionTruthStaleCount
#define twinrFsMissionTruthFlapCount twinrFs.missionTruthFlapCount
#define twinrFsMissionRangeFreshRaw twinrFs.missionRangeFreshRaw
#define twinrFsMissionRangeRiseRaw twinrFs.missionRangeRiseRaw
#define twinrFsMissionFlowReadyRaw twinrFs.missionFlowReadyRaw
#define twinrFsMissionAttitudeQuietRaw twinrFs.missionAttitudeQuietRaw
#define twinrFsLowBatteryCount twinrFs.lowBatteryCount
#define twinrFsCriticalBatteryCount twinrFs.criticalBatteryCount
#define twinrFsClearanceCount twinrFs.clearanceCount
#define twinrFsPendingClearanceReason twinrFs.pendingClearanceReason
#define twinrFsFlightObserved twinrFs.flightObserved
#define twinrFsGroundAbortQuiet twinrFs.groundAbortQuiet
#define twinrFsFrontId twinrFs.frontId
#define twinrFsBackId twinrFs.backId
#define twinrFsLeftId twinrFs.leftId
#define twinrFsRightId twinrFs.rightId
#define twinrFsUpId twinrFs.upId
#define twinrFsDownId twinrFs.downId
#define twinrFsVbatMvId twinrFs.vbatMvId
#define twinrFsPmStateId twinrFs.pmStateId
#define twinrFsStateEstimateZId twinrFs.stateEstimateZId
#define twinrFsStateEstimateVxId twinrFs.stateEstimateVxId
#define twinrFsStateEstimateVyId twinrFs.stateEstimateVyId
#define twinrFsStateEstimateVzId twinrFs.stateEstimateVzId
#define twinrFsMotionSqualId twinrFs.motionSqualId
#define twinrFsStabilizerRollId twinrFs.stabilizerRollId
#define twinrFsStabilizerPitchId twinrFs.stabilizerPitchId
#define twinrFsStabilizerThrustId twinrFs.stabilizerThrustId
#define twinrFsLastStateEstimateVx twinrFs.lastStateEstimateVx
#define twinrFsLastStateEstimateVy twinrFs.lastStateEstimateVy
#define twinrFsLastStateEstimateVz twinrFs.lastStateEstimateVz
#define twinrFsLastRollDeg twinrFs.lastRollDeg
#define twinrFsLastPitchDeg twinrFs.lastPitchDeg
#define twinrFsLastCommandedVx twinrFs.lastCommandedVx
#define twinrFsLastCommandedVy twinrFs.lastCommandedVy
#define twinrFsVerticalControl twinrFs.verticalControl
#define twinrFsHoverThrustEstimate twinrFs.verticalControl.hoverThrustEstimate
#define twinrFsFilteredBatteryMv twinrFs.verticalControl.filteredBatteryMv
#define twinrFsCeilingWithoutProgressAccumMs twinrFs.verticalControl.ceilingWithoutProgressAccumMs
#define twinrFsTakeoffProgressClass twinrFs.verticalControl.progressClass
#define twinrFsVerticalBatteryLimited twinrFs.verticalControl.batteryLimited
#define twinrFsLastObservedThrust twinrFs.lastObservedThrust
#define twinrFsDisturbanceControl twinrFs.disturbanceControl
#define twinrFsDisturbanceEstimateVx twinrFs.disturbanceControl.estimateVx
#define twinrFsDisturbanceEstimateVy twinrFs.disturbanceControl.estimateVy
#define twinrFsDisturbanceSeverityPermille twinrFs.disturbanceControl.severityPermille
#define twinrFsDisturbanceNonrecoverableCount twinrFs.disturbanceControl.nonrecoverableCount
#define twinrFsDisturbanceRecoverable twinrFs.disturbanceControl.recoverable
#define twinrFsDisturbanceSeverityClass twinrFs.disturbanceControl.severityClass
#define twinrFsDisturbanceNearGroundObserved twinrFs.disturbanceControl.nearGroundObserved
#define twinrFsTakeoffDebugFlags twinrFs.takeoffDebugFlags
#define twinrFsLateralCommandSource twinrFs.lateralCommandSource

static inline twinrFsVerticalControlConfig_t twinrFsVerticalDefaultConfig(void)
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
      .batteryLowMv = TWINR_FS_LOW_BATTERY_MV,
      .batteryCriticalMv = TWINR_FS_CRITICAL_BATTERY_MV,
      .hoverThrustEstAlpha = TWINR_FS_HOVER_THRUST_EST_ALPHA,
      .hoverThrustEstMaxDelta = TWINR_FS_HOVER_THRUST_EST_MAX_DELTA,
      .hoverThrustEstVzQuietMps = TWINR_FS_HOVER_THRUST_EST_VZ_QUIET_MPS,
      .hoverThrustEstZErrM = TWINR_FS_HOVER_THRUST_EST_Z_ERR_M,
      .batteryFilterAlpha = TWINR_FS_HOVER_BATTERY_FILTER_ALPHA,
  };
}

static inline twinrFsDisturbanceControlConfig_t twinrFsDisturbanceDefaultConfig(void)
{
  return (twinrFsDisturbanceControlConfig_t){
      .estimateAlpha = TWINR_FS_DISTURBANCE_ALPHA,
      .maxCommandMps = TWINR_FS_DISTURBANCE_MAX_MPS,
      .recoverableMagnitudeMps = TWINR_FS_DISTURBANCE_RECOVERABLE_MAX_MPS,
      .nonrecoverableSamples = TWINR_FS_DISTURBANCE_ABORT_SAMPLES,
  };
}

uint16_t clampUint16(uint32_t value);
uint16_t clampUint16Range(uint16_t value, uint16_t lower, uint16_t upper);
uint8_t clampUint8Range(uint8_t value, uint8_t lower, uint8_t upper);
float clampFloat(float value, float lower, float upper);
bool rangeIsValid(uint16_t rangeMm);
uint16_t readRangeMm(logVarId_t id);
uint16_t readVbatMv(void);
uint8_t readPmState(void);
float readStateEstimateZ(void);
float readStateEstimateVelocity(logVarId_t id);
float readStabilizerScalar(logVarId_t id);
uint16_t readMotionSqual(void);
void resetTriggerCounters(void);
void resetBatteryCounters(void);
void resetMissionState(void);
uint16_t stateEstimateZToMm(float stateEstimateZ);
uint8_t currentMissionFlags(bool airborne, bool rangeLive, bool flowLive);
void updateMissionTruthCounters(uint16_t downMm, uint16_t motionSqual);
bool missionRangeLive(void);
bool missionFlowLive(void);
bool missionAttitudeQuiet(void);
bool missionTruthStale(void);
bool missionStateFlapping(void);
bool flightObserved(bool isFlying, uint16_t downMm, float stateEstimateZ);
void bindSession(uint16_t sessionId);
void noteReject(twinrFsReject_t code, uint8_t version, uint8_t packetKind, uint16_t sessionId, uint16_t detail);
void maybeSendStatus(TickType_t now, bool force, bool airborne, bool rangeLive, bool flowLive, float stateEstimateZ);
void setState(twinrFsState_t state,
              twinrFsReason_t reason,
              TickType_t now,
              bool forceStatus,
              bool airborne,
              bool rangeLive,
              bool flowLive,
              float stateEstimateZ);
void recordTriggerEvent(twinrFsReason_t reason);
void recordLandedEvent(uint16_t downMm);
void sendFailsafeSetpoint(float vx,
                          float vy,
                          float z,
                          twinrFsLateralCommandSource_t source);
void sendManualTakeoffSetpoint(float thrust, twinrFsLateralCommandSource_t source);
uint16_t minClearanceMm(uint16_t frontMm,
                        uint16_t backMm,
                        uint16_t leftMm,
                        uint16_t rightMm,
                        uint16_t upMm);
bool batteryViolated(uint16_t vbatMv, uint8_t pmState, twinrFsReason_t *reason);
bool clearanceDebounced(uint16_t frontMm,
                        uint16_t backMm,
                        uint16_t leftMm,
                        uint16_t rightMm,
                        uint16_t upMm,
                        bool lateralClearanceArmed,
                        twinrFsReason_t *reason);
bool heartbeatExpired(TickType_t now);
void triggerFailsafe(twinrFsReason_t reason, TickType_t now, float stateEstimateZ, uint16_t downMm);
void processAppchannelPackets(TickType_t now, bool airborne, uint16_t downMm, float stateEstimateZ);
void runFailsafeControl(TickType_t now,
                        float stateEstimateZ,
                        uint16_t frontMm,
                        uint16_t backMm,
                        uint16_t leftMm,
                        uint16_t rightMm,
                        uint16_t downMm,
                        uint16_t motionSqual,
                        bool isFlying);
void runMissionControl(TickType_t now,
                       float stateEstimateZ,
                       uint16_t downMm,
                       uint16_t motionSqual,
                       bool isFlying);
void runTwinrFsStateMachine(TickType_t now, const twinrFsObservation_t *observation);

#endif
