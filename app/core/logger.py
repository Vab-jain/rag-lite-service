import sys
from loguru import logger

# Remove default handler
logger.remove()

# Add a standard handler with structured output format for production, human-readable for dev
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# You can also add file logging if needed
# logger.add("logs/app.log", rotation="10 MB", level="DEBUG")
