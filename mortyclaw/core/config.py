import os
from dotenv import load_dotenv

load_dotenv()

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(CORE_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

WORKSPACE_DIR = os.getenv(
    "MORTYCLAW_WORKSPACE",
    os.path.join(PROJECT_ROOT, "workspace")
)


DB_PATH = os.path.join(WORKSPACE_DIR, "state.sqlite3")     # 状态机：潜意识与短期记忆
RUNTIME_DB_PATH = os.path.join(WORKSPACE_DIR, "runtime.sqlite3")
MEMORY_DIR = os.path.join(WORKSPACE_DIR, "memory")         # 结构化记忆 + 人类可读快照
MEMORY_DB_PATH = os.path.join(MEMORY_DIR, "memory.sqlite3")
PERSONAS_DIR = os.path.join(WORKSPACE_DIR, "personas")     # 人设区：系统 Prompt
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")       # 脚本区：自动化武器库
OFFICE_DIR = os.path.join(WORKSPACE_DIR, "office")         # 沙盒工位 唯一被允许执行文件与shell操作的空间
SKILLS_DIR = os.path.join(OFFICE_DIR, "skills")            # 技能卡槽
TASKS_FILE = os.path.join(WORKSPACE_DIR, "tasks.json")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOGS_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "logs_archive")
BACKUPS_DIR = os.path.join(WORKSPACE_DIR, "backups")
RUNTIME_ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "runtime", "artifacts")

for d in [WORKSPACE_DIR, MEMORY_DIR, PERSONAS_DIR, SCRIPTS_DIR, OFFICE_DIR, SKILLS_DIR, LOGS_DIR, RUNTIME_ARTIFACTS_DIR]:
    os.makedirs(d, exist_ok=True)
