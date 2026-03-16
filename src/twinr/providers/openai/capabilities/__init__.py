"""Export the reusable capability mixins for Twinr's OpenAI backend package.

Import mixins from this package when composing backend surfaces such as
``OpenAIBackend``. The individual modules keep capability-specific behavior
split by concern: responses, search, speech, phrasing, and printing.
"""

from .phrasing import OpenAIMessagePhrasingMixin
from .printing import OpenAIPrintMixin
from .responses import OpenAIResponseMixin
from .search import OpenAISearchMixin
from .speech import OpenAISpeechMixin

__all__ = [
    "OpenAIMessagePhrasingMixin",
    "OpenAIPrintMixin",
    "OpenAIResponseMixin",
    "OpenAISearchMixin",
    "OpenAISpeechMixin",
]
