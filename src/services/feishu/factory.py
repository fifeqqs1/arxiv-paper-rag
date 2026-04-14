import logging
from typing import Optional

from src.config import get_settings
from src.db.factory import make_database
from src.services.cache.factory import make_redis_client
from src.services.feishu.bot import FeishuBot

logger = logging.getLogger(__name__)


def make_feishu_bot() -> Optional[FeishuBot]:
    """Create Feishu bot if enabled and configured."""
    settings = get_settings()

    if not settings.feishu.enabled:
        logger.info("Feishu bot is disabled")
        return None

    if not settings.feishu.app_id or not settings.feishu.app_secret:
        logger.warning("Feishu app credentials are not configured")
        return None

    redis_client = None
    try:
        redis_client = make_redis_client(settings)
    except Exception as exc:
        logger.warning(f"Feishu bot will start without Redis dedupe: {exc}")

    database = None
    try:
        database = make_database()
    except Exception as exc:
        logger.warning(f"Feishu bot will start without PostgreSQL conversation memory: {exc}")

    return FeishuBot(settings=settings, redis_client=redis_client, database=database)
