from form_agent.readers.base import Reader, available_readers, get_reader, register

# Import backends so they register themselves; skip ones whose SDK isn't installed.
try:
    from form_agent.readers import openai_reader  # noqa: F401
except ImportError:  # pragma: no cover
    pass
from form_agent.readers import hybrid  # noqa: F401,E402

__all__ = ["Reader", "available_readers", "get_reader", "register"]
