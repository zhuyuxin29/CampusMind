"""CampusMind 统一配置；修改后重启应用。密钥只从环境或根目录 .env 读取。"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
DATA_DIR = PROJECT_DIR / "new_data"
PDF_PATH = DATA_DIR / "raw" / "湖南信息学院学生手册.pdf"
PROCESSED_DIR = DATA_DIR / "processed"
STRUCTURED_DIR = PROCESSED_DIR / "structured_v2"
OCR_DIR = PROCESSED_DIR / "ocr_pages"
CHROMA_DIR = DATA_DIR / "chroma_db"
INDEX_ROOT = PROCESSED_DIR / "index_versions"

load_dotenv(ENV_PATH, override=False)
COLLECTION_NAME = os.getenv("CAMPUSMIND_COLLECTION", "campus_handbook_bge_m3_v2")
LEGACY_COLLECTION_NAME = "campus_handbook_bge_m3_v1"
SPLIT_VERSION = "scholarship_parent_child_v2"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
BATCH_SIZE = 16
MAX_QUESTION_CHARS = 1000
MAX_QUERIES = 3  # 完整问题 + 最多两条改写
RECALL_K = 6
CANDIDATE_K = 10
RRF_CONSTANT = 60
CONTEXT_TOP_N = 3
MAX_CONTEXT_CHARS = 6000
HISTORY_ROUNDS = 3
CONNECT_TIMEOUT = 10
REQUEST_TIMEOUT = 60
LLM_MAX_RETRIES = 1
EMBEDDING_MAX_RETRIES = 2
RERANK_MAX_RETRIES = 1
QUERY_MAX_TOKENS = 1000
ANSWER_MAX_TOKENS = 2000
# 引用校验失败时最多生成几次；一次正常生成 + 一次带纠正提示的重试。
ANSWER_ATTEMPTS = 2
# 引用句与原文去掉引用编号后的字符二元组最小重叠比例。
CITATION_SUPPORT_MIN_OVERLAP = 0.15


def get_api_key(name: str) -> str:
    """按需读取，不在日志或界面展示密钥。"""
    if name not in {"DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY"}:
        raise ValueError("不支持的密钥配置名称。")
    load_dotenv(ENV_PATH, override=False)
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"未配置 {name}，请检查项目根目录 .env。")
    return value


def missing_api_keys() -> list[str]:
    return [name for name in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY")
            if not os.getenv(name, "").strip()]
