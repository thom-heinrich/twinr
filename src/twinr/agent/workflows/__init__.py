from twinr.proactive import SocialTriggerDecision, SocialTriggerEngine
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.agent.workflows.runner import TwinrHardwareLoop

__all__ = [
    "SocialTriggerDecision",
    "SocialTriggerEngine",
    "TwinrHardwareLoop",
    "TwinrRealtimeHardwareLoop",
    "TwinrStreamingHardwareLoop",
]


def __getattr__(name: str):
    if name == "TwinrStreamingHardwareLoop":
        from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop

        return TwinrStreamingHardwareLoop
    raise AttributeError(name)
