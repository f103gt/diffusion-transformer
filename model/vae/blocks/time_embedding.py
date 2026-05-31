# Re-export from the shared model helpers module so that any code that
# imports directly from model.blocks.time_embedding continues to work.
from model.helpers import get_time_embedding  # noqa: F401
