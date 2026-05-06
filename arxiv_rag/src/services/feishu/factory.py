import logging
from typing import Any, Optional

import redis
from src.config import get_settings
from src.db.factory import make_database
from src.services.cache.factory import make_redis_client
from src.services.feishu.bot import FeishuBot, FeishuLocalRuntime

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


def make_local_feishu_bot(
    *,
    settings=None,
    redis_client: Optional[redis.Redis] = None,
    database: Optional[Any] = None,
    local_runtime: Optional[FeishuLocalRuntime] = None,
) -> FeishuBot:
    """Create a transport-free Feishu bot for local API callers."""
    settings = settings or get_settings()

    if redis_client is None:
        try:
            redis_client = make_redis_client(settings)
        except Exception as exc:
            logger.warning(f"Local Feishu bot will start without Redis conversation cache: {exc}")

    if database is None:
        try:
            database = make_database()
        except Exception as exc:
            logger.warning(f"Local Feishu bot will start without PostgreSQL conversation memory: {exc}")

    bot_kwargs = {
        "settings": settings,
        "redis_client": redis_client,
        "database": database,
        "build_client": False,
    }
    if local_runtime is not None:
        bot_kwargs["local_runtime"] = local_runtime

    return FeishuBot(**bot_kwargs)
