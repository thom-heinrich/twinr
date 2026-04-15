#include "twinr_on_device_failsafe_internal.h"

EVENTTRIGGER(twinrFsReject,
             uint8, rejectCode,
             uint8, version,
             uint8, packetKind,
             uint16, sessionId,
             uint16, detail)

EVENTTRIGGER(twinrFsOverflow,
             uint16, sessionId,
             uint16, heartbeatAgeMs)

static void sanitizeHeartbeatConfig(twinrFsHeartbeatConfig_t *config)
{
  config->heartbeatTimeoutMs = clampUint16Range(config->heartbeatTimeoutMs,
                                                TWINR_FS_HEARTBEAT_TIMEOUT_MIN_MS,
                                                TWINR_FS_HEARTBEAT_TIMEOUT_MAX_MS);
  config->lowBatteryMv = clampUint16Range(config->lowBatteryMv,
                                          TWINR_FS_BATTERY_THRESHOLD_MIN_MV,
                                          TWINR_FS_BATTERY_THRESHOLD_MAX_MV);
  config->criticalBatteryMv = clampUint16Range(config->criticalBatteryMv,
                                               TWINR_FS_BATTERY_THRESHOLD_MIN_MV,
                                               TWINR_FS_BATTERY_THRESHOLD_MAX_MV);
  if (config->criticalBatteryMv >= config->lowBatteryMv) {
    config->criticalBatteryMv = (config->lowBatteryMv > 100U)
                                  ? (uint16_t)(config->lowBatteryMv - 100U)
                                  : config->lowBatteryMv;
  }

  config->minClearanceMm = clampUint16Range(config->minClearanceMm,
                                            TWINR_FS_CLEARANCE_MIN_MM,
                                            TWINR_FS_CLEARANCE_MAX_MM);
  config->minUpClearanceMm = clampUint16Range(config->minUpClearanceMm,
                                              TWINR_FS_CLEARANCE_MIN_MM,
                                              TWINR_FS_CLEARANCE_MAX_MM);
  config->descentRateMmps = clampUint16Range(config->descentRateMmps,
                                             TWINR_FS_DESCENT_RATE_MIN_MMPS,
                                             TWINR_FS_DESCENT_RATE_MAX_MMPS);
  config->maxRepelVelocityMmps = clampUint16Range(config->maxRepelVelocityMmps,
                                                  0U,
                                                  TWINR_FS_MAX_REPEL_VELOCITY_LIMIT_MMPS);
  config->brakeHoldMs = clampUint16Range(config->brakeHoldMs, 0U, TWINR_FS_BRAKE_HOLD_MAX_MS);
  config->lowBatteryDebounceTicks = clampUint8Range(config->lowBatteryDebounceTicks,
                                                    TWINR_FS_DEBOUNCE_TICKS_MIN,
                                                    TWINR_FS_DEBOUNCE_TICKS_MAX);
  config->criticalBatteryDebounceTicks = clampUint8Range(config->criticalBatteryDebounceTicks,
                                                         TWINR_FS_DEBOUNCE_TICKS_MIN,
                                                         TWINR_FS_DEBOUNCE_TICKS_MAX);
  config->clearanceDebounceTicks = clampUint8Range(config->clearanceDebounceTicks,
                                                   TWINR_FS_DEBOUNCE_TICKS_MIN,
                                                   TWINR_FS_DEBOUNCE_TICKS_MAX);
}

static twinrFsHeartbeatConfig_t currentHeartbeatConfig(void)
{
  twinrFsHeartbeatConfig_t config = {
    .version = twinrFsPacketVersionInUse,
    .flags = (uint8_t)((twinrFsEnable ? TWINR_FS_FLAG_ENABLE : 0U)
                     | (twinrFsRequireClearance ? TWINR_FS_FLAG_REQUIRE_CLEARANCE : 0U)
                     | (twinrFsArmLateralClearance ? TWINR_FS_FLAG_ARM_LATERAL_CLEARANCE : 0U)),
    .sessionId = twinrFsSessionId,
    .heartbeatTimeoutMs = twinrFsHeartbeatTimeoutMs,
    .lowBatteryMv = twinrFsLowBatteryMv,
    .criticalBatteryMv = twinrFsCriticalBatteryMv,
    .minClearanceMm = twinrFsMinClearanceMm,
    .minUpClearanceMm = twinrFsMinUpClearanceMm,
    .descentRateMmps = twinrFsDescentRateMmps,
    .maxRepelVelocityMmps = twinrFsMaxRepelVelocityMmps,
    .brakeHoldMs = twinrFsBrakeHoldMs,
    .lowBatteryDebounceTicks = twinrFsLowBatteryDebounceTicks,
    .criticalBatteryDebounceTicks = twinrFsCriticalBatteryDebounceTicks,
    .clearanceDebounceTicks = twinrFsClearanceDebounceTicks,
    .hasSequence = false,
    .sequence = 0U,
  };
  return config;
}

static bool heartbeatConfigEqualsCurrent(const twinrFsHeartbeatConfig_t *config)
{
  const uint8_t currentFlags = (uint8_t)((twinrFsEnable ? TWINR_FS_FLAG_ENABLE : 0U)
                                       | (twinrFsRequireClearance ? TWINR_FS_FLAG_REQUIRE_CLEARANCE : 0U)
                                       | (twinrFsArmLateralClearance ? TWINR_FS_FLAG_ARM_LATERAL_CLEARANCE : 0U));

  return config->flags == currentFlags
      && config->heartbeatTimeoutMs == twinrFsHeartbeatTimeoutMs
      && config->lowBatteryMv == twinrFsLowBatteryMv
      && config->criticalBatteryMv == twinrFsCriticalBatteryMv
      && config->minClearanceMm == twinrFsMinClearanceMm
      && config->minUpClearanceMm == twinrFsMinUpClearanceMm
      && config->descentRateMmps == twinrFsDescentRateMmps
      && config->maxRepelVelocityMmps == twinrFsMaxRepelVelocityMmps
      && config->brakeHoldMs == twinrFsBrakeHoldMs
      && config->lowBatteryDebounceTicks == twinrFsLowBatteryDebounceTicks
      && config->criticalBatteryDebounceTicks == twinrFsCriticalBatteryDebounceTicks
      && config->clearanceDebounceTicks == twinrFsClearanceDebounceTicks;
}

static bool heartbeatConfigDiffersOnlyByLateralClearanceArm(const twinrFsHeartbeatConfig_t *config)
{
  const uint8_t comparableMask = (uint8_t)(TWINR_FS_FLAG_ENABLE | TWINR_FS_FLAG_REQUIRE_CLEARANCE);
  const uint8_t currentComparableFlags = (uint8_t)((twinrFsEnable ? TWINR_FS_FLAG_ENABLE : 0U)
                                                 | (twinrFsRequireClearance ? TWINR_FS_FLAG_REQUIRE_CLEARANCE : 0U));

  return (config->flags & comparableMask) == currentComparableFlags
      && config->heartbeatTimeoutMs == twinrFsHeartbeatTimeoutMs
      && config->lowBatteryMv == twinrFsLowBatteryMv
      && config->criticalBatteryMv == twinrFsCriticalBatteryMv
      && config->minClearanceMm == twinrFsMinClearanceMm
      && config->minUpClearanceMm == twinrFsMinUpClearanceMm
      && config->descentRateMmps == twinrFsDescentRateMmps
      && config->maxRepelVelocityMmps == twinrFsMaxRepelVelocityMmps
      && config->brakeHoldMs == twinrFsBrakeHoldMs
      && config->lowBatteryDebounceTicks == twinrFsLowBatteryDebounceTicks
      && config->criticalBatteryDebounceTicks == twinrFsCriticalBatteryDebounceTicks
      && config->clearanceDebounceTicks == twinrFsClearanceDebounceTicks;
}

void noteReject(const twinrFsReject_t code,
                const uint8_t version,
                const uint8_t packetKind,
                const uint16_t sessionId,
                const uint16_t detail)
{
  twinrFsLastRejectCode = (uint8_t)code;
  if (twinrFsRejectedPacketCount < UINT16_MAX) {
    twinrFsRejectedPacketCount += 1U;
  }

  eventTrigger_twinrFsReject_payload.rejectCode = (uint8_t)code;
  eventTrigger_twinrFsReject_payload.version = version;
  eventTrigger_twinrFsReject_payload.packetKind = packetKind;
  eventTrigger_twinrFsReject_payload.sessionId = sessionId;
  eventTrigger_twinrFsReject_payload.detail = detail;
  eventTrigger(&eventTrigger_twinrFsReject);

  DEBUG_PRINT("Reject packet code=%u ver=%u kind=%u session=%u detail=%u\n",
              (unsigned)code,
              (unsigned)version,
              (unsigned)packetKind,
              (unsigned)sessionId,
              (unsigned)detail);
}

static bool parseHeartbeatPacket(const uint8_t *buffer,
                                 const size_t length,
                                 twinrFsHeartbeatConfig_t *config)
{
  if (length < 2U) {
    noteReject(twinrFsRejectMalformed, 0U, 0U, 0U, (uint16_t)length);
    return false;
  }

  const uint8_t version = buffer[0];
  const uint8_t packetKind = buffer[1];

  if (packetKind != TWINR_FS_PACKET_KIND_HEARTBEAT) {
    noteReject(twinrFsRejectWrongKind, version, packetKind, 0U, (uint16_t)length);
    return false;
  }

  *config = currentHeartbeatConfig();
  config->version = version;

  if (version == TWINR_FS_PROTOCOL_VERSION_LEGACY) {
    if (length != sizeof(twinrFsHeartbeatPacketV1_t)) {
      noteReject(twinrFsRejectMalformed, version, packetKind, 0U, (uint16_t)length);
      return false;
    }

    twinrFsHeartbeatPacketV1_t packet;
    memcpy(&packet, buffer, sizeof(packet));

    config->flags = packet.flags;
    config->sessionId = packet.sessionId;
    if (packet.heartbeatTimeoutMs > 0U) {
      config->heartbeatTimeoutMs = packet.heartbeatTimeoutMs;
    }
    if (packet.lowBatteryMv > 0U) {
      config->lowBatteryMv = packet.lowBatteryMv;
    }
    if (packet.criticalBatteryMv > 0U) {
      config->criticalBatteryMv = packet.criticalBatteryMv;
    }
    if (packet.minClearanceMm > 0U) {
      config->minClearanceMm = packet.minClearanceMm;
    }
    if (packet.minUpClearanceMm > 0U) {
      config->minUpClearanceMm = packet.minUpClearanceMm;
    }
    if (packet.descentRateMmps > 0U) {
      config->descentRateMmps = packet.descentRateMmps;
    }
    if (packet.maxRepelVelocityMmps > 0U) {
      config->maxRepelVelocityMmps = packet.maxRepelVelocityMmps;
    }
    if (packet.brakeHoldMs > 0U) {
      config->brakeHoldMs = packet.brakeHoldMs;
    }
    config->hasSequence = false;
  } else if (version == TWINR_FS_PROTOCOL_VERSION_HEARTBEAT_V2) {
    if (length != sizeof(twinrFsHeartbeatPacketV2_t)) {
      noteReject(twinrFsRejectMalformed, version, packetKind, 0U, (uint16_t)length);
      return false;
    }

    twinrFsHeartbeatPacketV2_t packet;
    memcpy(&packet, buffer, sizeof(packet));

    config->flags = packet.flags;
    config->sessionId = packet.sessionId;
    if (packet.heartbeatTimeoutMs > 0U) {
      config->heartbeatTimeoutMs = packet.heartbeatTimeoutMs;
    }
    if (packet.lowBatteryMv > 0U) {
      config->lowBatteryMv = packet.lowBatteryMv;
    }
    if (packet.criticalBatteryMv > 0U) {
      config->criticalBatteryMv = packet.criticalBatteryMv;
    }
    if (packet.minClearanceMm > 0U) {
      config->minClearanceMm = packet.minClearanceMm;
    }
    if (packet.minUpClearanceMm > 0U) {
      config->minUpClearanceMm = packet.minUpClearanceMm;
    }
    if (packet.descentRateMmps > 0U) {
      config->descentRateMmps = packet.descentRateMmps;
    }
    if (packet.maxRepelVelocityMmps > 0U) {
      config->maxRepelVelocityMmps = packet.maxRepelVelocityMmps;
    }
    if (packet.brakeHoldMs > 0U) {
      config->brakeHoldMs = packet.brakeHoldMs;
    }
    if (packet.lowBatteryDebounceTicks > 0U) {
      config->lowBatteryDebounceTicks = packet.lowBatteryDebounceTicks;
    }
    if (packet.criticalBatteryDebounceTicks > 0U) {
      config->criticalBatteryDebounceTicks = packet.criticalBatteryDebounceTicks;
    }
    if (packet.clearanceDebounceTicks > 0U) {
      config->clearanceDebounceTicks = packet.clearanceDebounceTicks;
    }
    config->hasSequence = true;
    config->sequence = packet.sequence;
  } else {
    noteReject(twinrFsRejectUnsupportedVersion, version, packetKind, 0U, (uint16_t)length);
    return false;
  }

  sanitizeHeartbeatConfig(config);
  return true;
}

static bool parseCommandPacket(const uint8_t *buffer,
                               const size_t length,
                               twinrFsCommandPacket_t *command)
{
  if (length != sizeof(twinrFsCommandPacket_t)) {
    noteReject(twinrFsRejectMalformed, buffer[0], buffer[1], 0U, (uint16_t)length);
    return false;
  }

  memcpy(command, buffer, sizeof(twinrFsCommandPacket_t));
  if (command->packetKind != TWINR_FS_PACKET_KIND_COMMAND) {
    noteReject(twinrFsRejectWrongKind,
               command->version,
               command->packetKind,
               command->sessionId,
               (uint16_t)length);
    return false;
  }
  if (command->version != TWINR_FS_PROTOCOL_VERSION) {
    noteReject(twinrFsRejectUnsupportedVersion,
               command->version,
               command->packetKind,
               command->sessionId,
               (uint16_t)length);
    return false;
  }
  if (command->commandKind < TWINR_FS_HOVER_COMMAND_KIND_START
      || command->commandKind > TWINR_FS_HOVER_COMMAND_KIND_ABORT) {
    noteReject(twinrFsRejectWrongCommand,
               command->version,
               command->packetKind,
               command->sessionId,
               command->commandKind);
    return false;
  }

  command->targetHeightMm = clampUint16Range(command->targetHeightMm, 10U, 1200U);
  command->hoverDurationMs = clampUint16Range(command->hoverDurationMs, 100U, UINT16_MAX);
  command->takeoffRampMs = clampUint16Range(command->takeoffRampMs, 100U, 4000U);
  command->microHeightMm = clampUint16Range(command->microHeightMm, 40U, 400U);
  command->targetToleranceMm = clampUint16Range(command->targetToleranceMm, 10U, 200U);
  if (command->microHeightMm > command->targetHeightMm) {
    command->microHeightMm = command->targetHeightMm;
  }
  return true;
}

static void beginMission(const twinrFsCommandPacket_t *command,
                         const TickType_t now,
                         const uint16_t downMm,
                         const float stateEstimateZ)
{
  const uint16_t baselineHeightMm = rangeIsValid(downMm) ? downMm : stateEstimateZToMm(stateEstimateZ);
  resetMissionState();
  twinrFsMissionActive = 1U;
  twinrFsMissionTargetHeightMm = command->targetHeightMm;
  twinrFsMissionHoverDurationMs = command->hoverDurationMs;
  twinrFsMissionTakeoffRampMs = command->takeoffRampMs;
  twinrFsMissionMicroHeightMm = command->microHeightMm;
  twinrFsMissionTargetToleranceMm = command->targetToleranceMm;
  twinrFsMissionBaselineHeightMm = baselineHeightMm;
  twinrFsMissionCommandedHeightMm = baselineHeightMm;
  twinrFsMissionStartTick = now;
  twinrFsMissionPhaseStartTick = now;
  setState(twinrFsStateMissionTakeoff,
           twinrFsReasonNone,
           now,
           true,
           false,
           false,
           false,
           stateEstimateZ);
}

static void applyCommandPacket(const twinrFsCommandPacket_t *command,
                               const TickType_t now,
                               const bool airborne,
                               const uint16_t downMm,
                               const float stateEstimateZ)
{
  if (!twinrFsSessionBound || command->sessionId != twinrFsSessionId) {
    noteReject(twinrFsRejectWrongSessionCommand,
               command->version,
               command->packetKind,
               command->sessionId,
               0U);
    return;
  }

  twinrFsPacketVersionInUse = command->version;

  if (command->commandKind == TWINR_FS_HOVER_COMMAND_KIND_START) {
    if (twinrFsControlActive || twinrFsMissionActive || airborne) {
      noteReject(twinrFsRejectMissionBusy,
                 command->version,
                 command->packetKind,
                 command->sessionId,
                 twinrFsState);
      return;
    }
    beginMission(command, now, downMm, stateEstimateZ);
    return;
  }

  if (!twinrFsMissionActive) {
    noteReject(twinrFsRejectMissionBusy,
               command->version,
               command->packetKind,
               command->sessionId,
               twinrFsState);
    return;
  }

  if (command->commandKind == TWINR_FS_HOVER_COMMAND_KIND_LAND) {
    twinrFsMissionLandRequested = 1U;
    return;
  }

  resetMissionState();
  triggerFailsafe(twinrFsReasonMissionAbort, now, stateEstimateZ, downMm);
}

static void applyHeartbeatConfig(const twinrFsHeartbeatConfig_t *config,
                                 const TickType_t now,
                                 const bool airborne)
{
  const bool airborneOrActive = airborne || twinrFsControlActive || twinrFsMissionActive;
  const bool currentEnable = twinrFsEnable != 0U;
  const bool requestedEnable = (config->flags & TWINR_FS_FLAG_ENABLE) != 0U;
  const bool requestedArmLateralClearance =
      (config->flags & TWINR_FS_FLAG_ARM_LATERAL_CLEARANCE) != 0U;

  if (!twinrFsSessionBound) {
    bindSession(config->sessionId);
  } else if (config->sessionId != twinrFsSessionId) {
    if (airborneOrActive) {
      noteReject(twinrFsRejectSessionMismatch,
                 config->version,
                 TWINR_FS_PACKET_KIND_HEARTBEAT,
                 config->sessionId,
                 0U);
      return;
    }

    bindSession(config->sessionId);
  }

  if (config->hasSequence) {
    if (twinrFsHeartbeatSeqValid && config->sequence <= twinrFsLastHeartbeatSeq) {
      noteReject(twinrFsRejectStaleSequence,
                 config->version,
                 TWINR_FS_PACKET_KIND_HEARTBEAT,
                 config->sessionId,
                 (uint16_t)(config->sequence & 0xffffU));
      return;
    }
    twinrFsHeartbeatSeqValid = 1U;
    twinrFsLastHeartbeatSeq = config->sequence;
  } else if (twinrFsHeartbeatSeqValid && airborneOrActive) {
    noteReject(twinrFsRejectDowngrade,
               config->version,
               TWINR_FS_PACKET_KIND_HEARTBEAT,
               config->sessionId,
               0U);
    return;
  }

  twinrFsLastHeartbeatTick = now;
  twinrFsHeartbeatAgeMs = 0U;
  twinrFsPacketVersionInUse = config->version;

  if (airborneOrActive) {
    if ((config->flags & TWINR_FS_FLAG_ENABLE) == 0U) {
      noteReject(twinrFsRejectInFlightDisable,
                 config->version,
                 TWINR_FS_PACKET_KIND_HEARTBEAT,
                 config->sessionId,
                 0U);
    } else if (!heartbeatConfigEqualsCurrent(config)) {
      if (!twinrFsArmLateralClearance
          && requestedArmLateralClearance
          && heartbeatConfigDiffersOnlyByLateralClearanceArm(config)) {
        twinrFsArmLateralClearance = 1U;
        return;
      }
      noteReject(twinrFsRejectInFlightReconfigure,
                 config->version,
                 TWINR_FS_PACKET_KIND_HEARTBEAT,
                 config->sessionId,
                 0U);
    }
    return;
  }

  if (!requestedEnable || !currentEnable) {
    twinrFsFlightObserved = 0U;
    twinrFsGroundAbortQuiet = 0U;
  }

  twinrFsEnable = requestedEnable ? 1U : 0U;
  twinrFsRequireClearance = (config->flags & TWINR_FS_FLAG_REQUIRE_CLEARANCE) ? 1U : 0U;
  twinrFsArmLateralClearance = requestedArmLateralClearance ? 1U : 0U;
  twinrFsHeartbeatTimeoutMs = config->heartbeatTimeoutMs;
  twinrFsLowBatteryMv = config->lowBatteryMv;
  twinrFsCriticalBatteryMv = config->criticalBatteryMv;
  twinrFsMinClearanceMm = config->minClearanceMm;
  twinrFsMinUpClearanceMm = config->minUpClearanceMm;
  twinrFsDescentRateMmps = config->descentRateMmps;
  twinrFsMaxRepelVelocityMmps = config->maxRepelVelocityMmps;
  twinrFsBrakeHoldMs = config->brakeHoldMs;
  twinrFsLowBatteryDebounceTicks = config->lowBatteryDebounceTicks;
  twinrFsCriticalBatteryDebounceTicks = config->criticalBatteryDebounceTicks;
  twinrFsClearanceDebounceTicks = config->clearanceDebounceTicks;
  resetTriggerCounters();

  if (!twinrFsEnable && !twinrFsControlActive) {
    setState(twinrFsStateDisabled, twinrFsReasonManualDisable, now, true, airborne, false, false, 0.0f);
  } else if (!twinrFsControlActive) {
    setState(twinrFsStateMonitoring, twinrFsReasonNone, now, true, airborne, false, false, 0.0f);
  }
}

void processAppchannelPackets(const TickType_t now,
                              const bool airborne,
                              const uint16_t downMm,
                              const float stateEstimateZ)
{
  uint8_t buffer[APPCHANNEL_MTU];
  size_t length = 0U;

  while ((length = appchannelReceiveDataPacket(buffer, sizeof(buffer), 0)) > 0U) {
    if (length >= 2U && buffer[1] == TWINR_FS_PACKET_KIND_COMMAND) {
      twinrFsCommandPacket_t command;
      if (!parseCommandPacket(buffer, length, &command)) {
        continue;
      }
      applyCommandPacket(&command, now, airborne, downMm, stateEstimateZ);
      continue;
    }

    twinrFsHeartbeatConfig_t heartbeatConfig;
    if (!parseHeartbeatPacket(buffer, length, &heartbeatConfig)) {
      continue;
    }

    applyHeartbeatConfig(&heartbeatConfig, now, airborne);
  }

  if (appchannelHasOverflowOccurred()) {
    if (twinrFsOverflowCount < UINT16_MAX) {
      twinrFsOverflowCount += 1U;
    }

    eventTrigger_twinrFsOverflow_payload.sessionId = twinrFsSessionId;
    eventTrigger_twinrFsOverflow_payload.heartbeatAgeMs = twinrFsHeartbeatAgeMs;
    eventTrigger(&eventTrigger_twinrFsOverflow);

    DEBUG_PRINT("Appchannel RX overflow\n");
  }
}
