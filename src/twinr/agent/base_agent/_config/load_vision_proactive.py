"""Load camera topology, portrait matching, proactive sensing, and drone settings."""

from __future__ import annotations

from .context import ConfigLoadContext
from .parsing import (
    _parse_bool,
    _parse_clamped_float,
    _parse_float,
)


def load_vision_proactive_config(context: ConfigLoadContext) -> dict[str, object]:
    """Return the config fields owned by this loading domain."""

    get_value = context.get_value
    camera_host_mode = context.camera_host_mode
    effective_second_pi_base_url = context.effective_second_pi_base_url
    camera_device = context.camera_device
    camera_proxy_snapshot_url = context.camera_proxy_snapshot_url
    proactive_vision_provider = context.proactive_vision_provider
    drone_base_url = context.drone_base_url

    return {
        "camera_host_mode": camera_host_mode,
        "camera_second_pi_base_url": effective_second_pi_base_url,
        "camera_device": camera_device,
        "camera_width": int(get_value("TWINR_CAMERA_WIDTH", "640") or "640"),
        "camera_height": int(get_value("TWINR_CAMERA_HEIGHT", "480") or "480"),
        "camera_framerate": int(get_value("TWINR_CAMERA_FRAMERATE", "30") or "30"),
        "camera_input_format": get_value("TWINR_CAMERA_INPUT_FORMAT"),
        "camera_ffmpeg_path": get_value("TWINR_CAMERA_FFMPEG_PATH", "ffmpeg")
        or "ffmpeg",
        "camera_proxy_snapshot_url": camera_proxy_snapshot_url,
        "vision_reference_image_path": get_value("TWINR_VISION_REFERENCE_IMAGE"),
        "portrait_match_enabled": _parse_bool(
            get_value("TWINR_PORTRAIT_MATCH_ENABLED"), True
        ),
        "portrait_match_detector_model_path": get_value(
            "TWINR_PORTRAIT_MATCH_DETECTOR_MODEL_PATH",
            "state/opencv/models/face_detection_yunet_2023mar.onnx",
        )
        or "state/opencv/models/face_detection_yunet_2023mar.onnx",
        "portrait_match_recognizer_model_path": get_value(
            "TWINR_PORTRAIT_MATCH_RECOGNIZER_MODEL_PATH",
            "state/opencv/models/face_recognition_sface_2021dec.onnx",
        )
        or "state/opencv/models/face_recognition_sface_2021dec.onnx",
        "portrait_match_likely_threshold": _parse_clamped_float(
            get_value("TWINR_PORTRAIT_MATCH_LIKELY_THRESHOLD"),
            0.45,
            minimum=0.0,
            maximum=1.0,
        ),
        "portrait_match_uncertain_threshold": _parse_clamped_float(
            get_value("TWINR_PORTRAIT_MATCH_UNCERTAIN_THRESHOLD"),
            0.34,
            minimum=0.0,
            maximum=1.0,
        ),
        "portrait_match_max_age_s": _parse_float(
            get_value("TWINR_PORTRAIT_MATCH_MAX_AGE_S"), 45.0, minimum=0.0
        ),
        "portrait_match_capture_lock_timeout_s": _parse_float(
            get_value("TWINR_PORTRAIT_MATCH_CAPTURE_LOCK_TIMEOUT_S"), 5.0, minimum=0.0
        ),
        "portrait_match_store_path": get_value(
            "TWINR_PORTRAIT_MATCH_STORE_PATH", "state/portrait_identities.json"
        )
        or "state/portrait_identities.json",
        "portrait_match_reference_image_dir": get_value(
            "TWINR_PORTRAIT_MATCH_REFERENCE_IMAGE_DIR", "state/portrait_identities"
        )
        or "state/portrait_identities",
        "portrait_match_primary_user_id": get_value(
            "TWINR_PORTRAIT_MATCH_PRIMARY_USER_ID", "main_user"
        )
        or "main_user",
        "portrait_match_max_reference_images_per_user": max(
            1,
            int(
                get_value("TWINR_PORTRAIT_MATCH_MAX_REFERENCE_IMAGES_PER_USER", "6")
                or "6"
            ),
        ),
        "portrait_match_identity_margin": _parse_clamped_float(
            get_value("TWINR_PORTRAIT_MATCH_IDENTITY_MARGIN"),
            0.05,
            minimum=0.0,
            maximum=1.0,
        ),
        "portrait_match_temporal_window_s": _parse_float(
            get_value("TWINR_PORTRAIT_MATCH_TEMPORAL_WINDOW_S"), 300.0, minimum=0.0
        ),
        "portrait_match_temporal_min_observations": max(
            1,
            int(
                get_value("TWINR_PORTRAIT_MATCH_TEMPORAL_MIN_OBSERVATIONS", "2") or "2"
            ),
        ),
        "portrait_match_temporal_max_observations": max(
            1,
            int(
                get_value("TWINR_PORTRAIT_MATCH_TEMPORAL_MAX_OBSERVATIONS", "12")
                or "12"
            ),
        ),
        "openai_vision_detail": get_value("OPENAI_VISION_DETAIL", "auto") or "auto",
        "proactive_enabled": _parse_bool(get_value("TWINR_PROACTIVE_ENABLED"), False),
        "proactive_vision_provider": proactive_vision_provider,
        "proactive_remote_camera_base_url": effective_second_pi_base_url,
        "proactive_remote_camera_timeout_s": _parse_float(
            get_value("TWINR_PROACTIVE_REMOTE_CAMERA_TIMEOUT_S"), 4.0, minimum=0.1
        ),
        "proactive_local_camera_detection_network_path": get_value(
            "TWINR_PROACTIVE_LOCAL_CAMERA_DETECTION_NETWORK_PATH",
            "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
        )
        or "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
        "proactive_local_camera_pose_network_path": get_value(
            "TWINR_PROACTIVE_LOCAL_CAMERA_POSE_NETWORK_PATH",
            "/usr/share/imx500-models/imx500_network_posenet.rpk",
        )
        or "/usr/share/imx500-models/imx500_network_posenet.rpk",
        "proactive_local_camera_pose_backend": (
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_POSE_BACKEND", "mediapipe")
            or "mediapipe"
        )
        .strip()
        .lower(),
        "proactive_local_camera_mediapipe_pose_model_path": get_value(
            "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_POSE_MODEL_PATH",
            "state/mediapipe/models/pose_landmarker_full.task",
        )
        or "state/mediapipe/models/pose_landmarker_full.task",
        "proactive_local_camera_mediapipe_hand_landmarker_model_path": get_value(
            "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_HAND_LANDMARKER_MODEL_PATH",
            "state/mediapipe/models/hand_landmarker.task",
        )
        or "state/mediapipe/models/hand_landmarker.task",
        "proactive_local_camera_mediapipe_gesture_model_path": get_value(
            "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_GESTURE_MODEL_PATH",
            "state/mediapipe/models/gesture_recognizer.task",
        )
        or "state/mediapipe/models/gesture_recognizer.task",
        "proactive_local_camera_mediapipe_custom_gesture_model_path": get_value(
            "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_CUSTOM_GESTURE_MODEL_PATH"
        )
        or None,
        "proactive_local_camera_mediapipe_num_hands": int(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_NUM_HANDS", "2") or "2"
        ),
        "proactive_local_camera_sequence_window_s": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_SEQUENCE_WINDOW_S"), 0.55
        ),
        "proactive_local_camera_sequence_min_frames": int(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_SEQUENCE_MIN_FRAMES", "3") or "3"
        ),
        "proactive_local_camera_source_device": get_value(
            "TWINR_PROACTIVE_LOCAL_CAMERA_SOURCE_DEVICE", "imx500"
        )
        or "imx500",
        "proactive_local_camera_frame_rate": int(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_FRAME_RATE", "15") or "15"
        ),
        "proactive_local_camera_lock_timeout_s": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_LOCK_TIMEOUT_S"), 5.0
        ),
        "proactive_local_camera_startup_warmup_s": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_STARTUP_WARMUP_S"), 0.8
        ),
        "proactive_local_camera_metadata_wait_s": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_METADATA_WAIT_S"), 0.75
        ),
        "proactive_local_camera_person_confidence_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_PERSON_CONFIDENCE_THRESHOLD"), 0.4
        ),
        "proactive_local_camera_object_confidence_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_OBJECT_CONFIDENCE_THRESHOLD"), 0.55
        ),
        "proactive_local_camera_person_near_area_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_PERSON_NEAR_AREA_THRESHOLD"), 0.2
        ),
        "proactive_local_camera_person_near_height_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_PERSON_NEAR_HEIGHT_THRESHOLD"), 0.55
        ),
        "proactive_local_camera_object_near_area_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_OBJECT_NEAR_AREA_THRESHOLD"), 0.08
        ),
        "proactive_local_camera_attention_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_ATTENTION_SCORE_THRESHOLD"), 0.62
        ),
        "proactive_local_camera_engaged_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_ENGAGED_SCORE_THRESHOLD"), 0.45
        ),
        "proactive_local_camera_pose_confidence_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_POSE_CONFIDENCE_THRESHOLD"), 0.3
        ),
        "proactive_local_camera_pose_refresh_s": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_POSE_REFRESH_S"), 0.75
        ),
        "proactive_local_camera_builtin_gesture_min_score": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_BUILTIN_GESTURE_MIN_SCORE"), 0.35
        ),
        "proactive_local_camera_custom_gesture_min_score": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_CUSTOM_GESTURE_MIN_SCORE"), 0.45
        ),
        "proactive_local_camera_min_hand_detection_confidence": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_MIN_HAND_DETECTION_CONFIDENCE"),
            0.35,
        ),
        "proactive_local_camera_min_hand_presence_confidence": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_MIN_HAND_PRESENCE_CONFIDENCE"), 0.35
        ),
        "proactive_local_camera_min_hand_tracking_confidence": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_MIN_HAND_TRACKING_CONFIDENCE"), 0.35
        ),
        "proactive_local_camera_max_roi_candidates": int(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_MAX_ROI_CANDIDATES", "4") or "4"
        ),
        "proactive_local_camera_primary_person_roi_padding": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_PRIMARY_PERSON_ROI_PADDING"), 0.18
        ),
        "proactive_local_camera_primary_person_upper_body_ratio": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_PRIMARY_PERSON_UPPER_BODY_RATIO"),
            0.78,
        ),
        "proactive_local_camera_wrist_roi_scale": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_WRIST_ROI_SCALE"), 0.34
        ),
        "proactive_local_camera_fine_hand_explicit_hold_s": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_FINE_HAND_EXPLICIT_HOLD_S"), 0.45
        ),
        "proactive_local_camera_fine_hand_explicit_confirm_samples": int(
            get_value(
                "TWINR_PROACTIVE_LOCAL_CAMERA_FINE_HAND_EXPLICIT_CONFIRM_SAMPLES", "1"
            )
            or "1"
        ),
        "proactive_local_camera_fine_hand_explicit_min_confidence": _parse_float(
            get_value("TWINR_PROACTIVE_LOCAL_CAMERA_FINE_HAND_EXPLICIT_MIN_CONFIDENCE"),
            0.72,
        ),
        "gesture_wakeup_enabled": _parse_bool(
            get_value("TWINR_GESTURE_WAKEUP_ENABLED"), True
        ),
        "gesture_wakeup_trigger": (
            get_value("TWINR_GESTURE_WAKEUP_TRIGGER", "peace_sign") or "peace_sign"
        )
        .strip()
        .lower(),
        "gesture_wakeup_cooldown_s": _parse_float(
            get_value("TWINR_GESTURE_WAKEUP_COOLDOWN_S"), 3.0, minimum=0.0
        ),
        "proactive_poll_interval_s": _parse_float(
            get_value("TWINR_PROACTIVE_POLL_INTERVAL_S"), 4.0
        ),
        "proactive_capture_interval_s": _parse_float(
            get_value("TWINR_PROACTIVE_CAPTURE_INTERVAL_S"), 6.0
        ),
        "proactive_motion_window_s": _parse_float(
            get_value("TWINR_PROACTIVE_MOTION_WINDOW_S"), 20.0
        ),
        "proactive_low_motion_after_s": _parse_float(
            get_value("TWINR_PROACTIVE_LOW_MOTION_AFTER_S"), 12.0
        ),
        "proactive_audio_enabled": _parse_bool(
            get_value("TWINR_PROACTIVE_AUDIO_ENABLED"), False
        ),
        "proactive_audio_input_device": get_value("TWINR_PROACTIVE_AUDIO_INPUT_DEVICE")
        or get_value("TWINR_PROACTIVE_AUDIO_DEVICE"),
        "proactive_audio_sample_ms": int(
            get_value("TWINR_PROACTIVE_AUDIO_SAMPLE_MS", "1000") or "1000"
        ),
        "proactive_audio_distress_enabled": _parse_bool(
            get_value("TWINR_PROACTIVE_AUDIO_DISTRESS_ENABLED"), False
        ),
        "proactive_vision_review_enabled": _parse_bool(
            get_value("TWINR_PROACTIVE_VISION_REVIEW_ENABLED"), False
        ),
        "proactive_vision_review_buffer_frames": int(
            get_value("TWINR_PROACTIVE_VISION_REVIEW_BUFFER_FRAMES", "8") or "8"
        ),
        "proactive_vision_review_max_frames": int(
            get_value("TWINR_PROACTIVE_VISION_REVIEW_MAX_FRAMES", "4") or "4"
        ),
        "proactive_vision_review_max_age_s": _parse_float(
            get_value("TWINR_PROACTIVE_VISION_REVIEW_MAX_AGE_S"), 12.0
        ),
        "proactive_vision_review_min_spacing_s": _parse_float(
            get_value("TWINR_PROACTIVE_VISION_REVIEW_MIN_SPACING_S"), 1.2
        ),
        "proactive_person_returned_absence_s": _parse_float(
            get_value("TWINR_PROACTIVE_PERSON_RETURNED_ABSENCE_S"), 20.0 * 60.0
        ),
        "proactive_person_returned_recent_motion_s": _parse_float(
            get_value("TWINR_PROACTIVE_PERSON_RETURNED_RECENT_MOTION_S"), 30.0
        ),
        "proactive_attention_window_s": _parse_float(
            get_value("TWINR_PROACTIVE_ATTENTION_WINDOW_S"), 6.0
        ),
        "proactive_slumped_quiet_s": _parse_float(
            get_value("TWINR_PROACTIVE_SLUMPED_QUIET_S"), 20.0
        ),
        "proactive_possible_fall_stillness_s": _parse_float(
            get_value("TWINR_PROACTIVE_POSSIBLE_FALL_STILLNESS_S"), 10.0
        ),
        "proactive_possible_fall_visibility_loss_hold_s": _parse_float(
            get_value("TWINR_PROACTIVE_POSSIBLE_FALL_VISIBILITY_LOSS_HOLD_S"), 15.0
        ),
        "proactive_possible_fall_visibility_loss_arming_s": _parse_float(
            get_value("TWINR_PROACTIVE_POSSIBLE_FALL_VISIBILITY_LOSS_ARMING_S"), 6.0
        ),
        "proactive_possible_fall_slumped_visibility_loss_arming_s": _parse_float(
            get_value("TWINR_PROACTIVE_POSSIBLE_FALL_SLUMPED_VISIBILITY_LOSS_ARMING_S"),
            4.0,
        ),
        "proactive_possible_fall_once_per_presence_session": _parse_bool(
            get_value("TWINR_PROACTIVE_POSSIBLE_FALL_ONCE_PER_PRESENCE_SESSION"), True
        ),
        "proactive_floor_stillness_s": _parse_float(
            get_value("TWINR_PROACTIVE_FLOOR_STILLNESS_S"), 20.0
        ),
        "proactive_showing_intent_hold_s": _parse_float(
            get_value("TWINR_PROACTIVE_SHOWING_INTENT_HOLD_S"), 1.5
        ),
        "proactive_positive_contact_hold_s": _parse_float(
            get_value("TWINR_PROACTIVE_POSITIVE_CONTACT_HOLD_S"), 1.5
        ),
        "proactive_distress_hold_s": _parse_float(
            get_value("TWINR_PROACTIVE_DISTRESS_HOLD_S"), 3.0
        ),
        "proactive_fall_transition_window_s": _parse_float(
            get_value("TWINR_PROACTIVE_FALL_TRANSITION_WINDOW_S"), 12.0
        ),
        "proactive_person_returned_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_PERSON_RETURNED_SCORE_THRESHOLD"), 0.9
        ),
        "proactive_attention_window_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_ATTENTION_WINDOW_SCORE_THRESHOLD"), 0.86
        ),
        "proactive_slumped_quiet_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_SLUMPED_QUIET_SCORE_THRESHOLD"), 0.9
        ),
        "proactive_possible_fall_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_POSSIBLE_FALL_SCORE_THRESHOLD"), 0.82
        ),
        "proactive_floor_stillness_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_FLOOR_STILLNESS_SCORE_THRESHOLD"), 0.9
        ),
        "proactive_showing_intent_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_SHOWING_INTENT_SCORE_THRESHOLD"), 0.84
        ),
        "proactive_positive_contact_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_POSITIVE_CONTACT_SCORE_THRESHOLD"), 0.84
        ),
        "proactive_distress_possible_score_threshold": _parse_float(
            get_value("TWINR_PROACTIVE_DISTRESS_POSSIBLE_SCORE_THRESHOLD"), 0.85
        ),
        "proactive_governor_enabled": _parse_bool(
            get_value("TWINR_PROACTIVE_GOVERNOR_ENABLED"), True
        ),
        "proactive_governor_active_reservation_ttl_s": _parse_float(
            get_value("TWINR_PROACTIVE_GOVERNOR_ACTIVE_RESERVATION_TTL_S"), 45.0
        ),
        "proactive_governor_global_prompt_cooldown_s": _parse_float(
            get_value("TWINR_PROACTIVE_GOVERNOR_GLOBAL_PROMPT_COOLDOWN_S"), 120.0
        ),
        "proactive_governor_window_s": _parse_float(
            get_value("TWINR_PROACTIVE_GOVERNOR_WINDOW_S"), 20.0 * 60.0
        ),
        "proactive_governor_window_prompt_limit": int(
            get_value("TWINR_PROACTIVE_GOVERNOR_WINDOW_PROMPT_LIMIT", "4") or "4"
        ),
        "proactive_governor_presence_session_prompt_limit": int(
            get_value("TWINR_PROACTIVE_GOVERNOR_PRESENCE_SESSION_PROMPT_LIMIT", "2")
            or "2"
        ),
        "proactive_governor_source_repeat_cooldown_s": _parse_float(
            get_value("TWINR_PROACTIVE_GOVERNOR_SOURCE_REPEAT_COOLDOWN_S"), 10.0 * 60.0
        ),
        "proactive_governor_history_limit": int(
            get_value("TWINR_PROACTIVE_GOVERNOR_HISTORY_LIMIT", "128") or "128"
        ),
        "proactive_visual_first_audio_global_cooldown_s": _parse_float(
            get_value("TWINR_PROACTIVE_VISUAL_FIRST_AUDIO_GLOBAL_COOLDOWN_S"),
            5.0 * 60.0,
        ),
        "proactive_visual_first_audio_source_repeat_cooldown_s": _parse_float(
            get_value("TWINR_PROACTIVE_VISUAL_FIRST_AUDIO_SOURCE_REPEAT_COOLDOWN_S"),
            15.0 * 60.0,
        ),
        "proactive_visual_first_cue_hold_s": _parse_float(
            get_value("TWINR_PROACTIVE_VISUAL_FIRST_CUE_HOLD_S"), 45.0
        ),
        "proactive_quiet_hours_visual_only_enabled": _parse_bool(
            get_value("TWINR_PROACTIVE_QUIET_HOURS_VISUAL_ONLY_ENABLED"), True
        ),
        "proactive_quiet_hours_start_local": get_value(
            "TWINR_PROACTIVE_QUIET_HOURS_START_LOCAL", "21:00"
        )
        or "21:00",
        "proactive_quiet_hours_end_local": get_value(
            "TWINR_PROACTIVE_QUIET_HOURS_END_LOCAL", "07:00"
        )
        or "07:00",
        "drone_enabled": _parse_bool(get_value("TWINR_DRONE_ENABLED"), False),
        "drone_base_url": drone_base_url,
        "drone_require_manual_arm": _parse_bool(
            get_value("TWINR_DRONE_REQUIRE_MANUAL_ARM"), True
        ),
        "drone_mission_timeout_s": _parse_float(
            get_value("TWINR_DRONE_MISSION_TIMEOUT_S"), 45.0
        ),
        "drone_request_timeout_s": _parse_float(
            get_value("TWINR_DRONE_REQUEST_TIMEOUT_S"), 2.0
        ),
    }
