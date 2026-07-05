"""
全局配置常量
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent

# SQLite 数据库文件路径（存放在项目根目录）
DATABASE_URL = f"sqlite:///{BASE_DIR / 'chat.db'}"

# 每次拼给大模型的历史消息条数上限（user + assistant 成对计）
HISTORY_LIMIT = 10

# 内存缓存过期时间（秒），30 分钟未活跃的会话从缓存中移除
CACHE_TTL = 1800
