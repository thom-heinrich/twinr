"""Reject retired helper-Pi hardware topologies during config normalization.

Twinr's productive hardware layout is now single-Pi only: the still/proactive
camera and the attention-servo output must both live on the main runtime Pi.
Legacy helper-Pi camera and servo envs remain in the repo only as historical
artifacts; the canonical config loader must fail closed when those removed
topologies are still configured.
"""

from __future__ import annotations


def validate_supported_camera_topology(
    *,
    camera_host_mode: str,
    camera_second_pi_base_url: str | None,
    camera_proxy_snapshot_url: str | None,
    proactive_remote_camera_base_url: str | None,
) -> None:
    """Reject removed helper-Pi camera settings with one operator-facing error.

    Args:
        camera_host_mode: Normalized high-level camera topology mode.
        camera_second_pi_base_url: Optional legacy helper-Pi base URL.
        camera_proxy_snapshot_url: Optional legacy snapshot-proxy URL.
        proactive_remote_camera_base_url: Optional legacy proactive helper URL.

    Raises:
        ValueError: If any retired helper-Pi camera setting is still present.
    """

    detected: list[str] = []
    if camera_host_mode == "second_pi":
        detected.append("TWINR_CAMERA_HOST_MODE=second_pi")
    if camera_second_pi_base_url:
        detected.append("TWINR_CAMERA_SECOND_PI_BASE_URL")
    if camera_proxy_snapshot_url:
        detected.append("TWINR_CAMERA_PROXY_SNAPSHOT_URL")
    if proactive_remote_camera_base_url:
        detected.append("TWINR_PROACTIVE_REMOTE_CAMERA_BASE_URL")
    if not detected:
        return
    raise ValueError(
        "Legacy helper-Pi camera topology is no longer supported. "
        "Twinr now requires the camera on the main Pi. "
        "Remove the retired camera envs: "
        + ", ".join(detected)
        + "."
    )


def validate_supported_attention_servo_topology(
    *,
    attention_servo_driver: str,
    attention_servo_peer_base_url: str | None,
) -> None:
    """Reject removed helper-Pi servo settings with one operator-facing error.

    Args:
        attention_servo_driver: Normalized attention-servo driver identifier.
        attention_servo_peer_base_url: Optional legacy helper-Pi servo proxy URL.

    Raises:
        ValueError: If any retired helper-Pi servo setting is still present.
    """

    detected: list[str] = []
    if attention_servo_driver == "peer_pololu_maestro":
        detected.append("TWINR_ATTENTION_SERVO_DRIVER=peer_pololu_maestro")
    if attention_servo_peer_base_url:
        detected.append("TWINR_ATTENTION_SERVO_PEER_BASE_URL")
    if not detected:
        return
    raise ValueError(
        "Legacy helper-Pi attention-servo topology is no longer supported. "
        "Twinr now requires direct servo output on the main Pi. "
        "Remove the retired servo envs: "
        + ", ".join(detected)
        + "."
    )
