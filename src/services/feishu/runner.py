import logging

from src.services.feishu.factory import make_feishu_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Start the standalone Feishu bot process."""
    bot = make_feishu_bot()
    if not bot:
        logger.warning("Feishu bot is not enabled or not fully configured")
        return 0

    bot.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
