"""
DriveSmart AI - Modern Chatbot Interface (High-Tech)
Chat-first UI with Security + Analytics + Supported States
Save as: dashboard.py
"""

import streamlit as st
import time
import os
import sys
from pathlib import Path
import sqlite3
from dotenv import load_dotenv
import uuid
import inspect
import re
import html
import json
import base64


# -------------------- ENV LOADER --------------------
def load_env():
    candidates = [
        Path(__file__).parent / ".env",
        Path(__file__).parent / "SmartDrive" / ".env",
        Path(__file__).parent / "SmartDrive" / "src" / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=True)
            return p
    return None

load_env()

# Background image base64 (optional). Use env var `BG_B64` if provided, else empty.
BG_B64 = os.getenv("BG_B64", "") or ""

# -------------------- PATH SETUP --------------------
ROOT = Path(__file__).resolve().parent
# Ensure repository root and package directories are on sys.path so
# `import SmartDrive...` works both locally and on Streamlit Cloud.
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "SmartDrive"))

# -------------------- SECURITY MODULES --------------------
# Be defensive: catch any exception during imports (KeyError can occur
# inside import machinery in some environments). We don't want the app
# to crash before rendering static UI like the intro image.
try:
    from SmartDrive.security.input_validator import PromptSecurityValidator
    from SmartDrive.security.output_validator import ResponseValidator
    from SmartDrive.security.behavioral_monitor import BehavioralMonitor
    SECURITY_ENABLED = True
except Exception:
    SECURITY_ENABLED = False

# -------------------- CORE MODULES --------------------
# Defensive import: catch any Exception (not just ImportError) so the
# app can continue rendering static UI even if package imports fail
# inside the Streamlit Cloud import system.
try:
    from SmartDrive.src.refined_prompts import RefinedDriveSmartWorkflow
    from SmartDrive.src.vector_store import CloudTrafficLawVectorStore
    MODULES_LOADED = True
except Exception:
    MODULES_LOADED = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="DriveSmart AI Chat",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Inject Material Icons / Material Symbols so ligature text like
# "<" renders as an icon instead of raw text.
try:
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0" rel="stylesheet" />
        <style>
        .material-icons, .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 48; }
        </style>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    # If injection fails, we still want the app to continue
    pass
# -------------------- SCOPE GUARD --------------------
TRAFFIC_KEYWORDS = {
    "traffic", "drive", "driving", "driver", "license", "licence",
    "road", "highway", "street", "intersection", "signal", "stop sign",
    "red light", "green light", "yellow light",
    "speed", "speeding", "limit", "school zone",
    "dui", "dwi", "alcohol", "impaired",
    "seat belt", "seatbelt", "phone", "texting", "handsfree",
    "parking", "registration", "insurance",
    "vehicle", "car", "motorcycle", "truck",
    "ticket", "fine", "points", "suspension", "reckless"
}

OUT_OF_SCOPE_MESSAGE = (
    "I’m DriveSmart AI — a traffic-law assistant. "
    "I provide guidance on traffic rules, penalties, and safe driving across supported states. "
    "Your question looks outside my scope. "
    "Please ask me something related to traffic laws or safe driving in a supported state."
)

def is_traffic_related(query: str) -> bool:
    if not query:
        return False
    q = query.lower().strip()

    # quick early allow for common traffic intents
    if any(x in q for x in ["speed limit", "turn right", "license", "dui", "parking", "red light"]):
        return True

    return any(k in q for k in TRAFFIC_KEYWORDS)

def build_out_of_scope_answer(supported_states):
    # optional: include a small states hint without making this long
    preview = ", ".join((supported_states or [])[:5])
    if preview:
        return (
            "I’m DriveSmart AI — a traffic-law assistant. "
            f"I provide guidance on traffic rules, penalties, and safe driving across {preview} and other supported states. "
            "Your question looks outside my scope. "
            "Please ask me something related to traffic laws or safe driving in a supported state."
        )
    return OUT_OF_SCOPE_MESSAGE
# -------------------- SESSION STATE (ONLY ONCE) --------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "processing" not in st.session_state:
    st.session_state.processing = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I’m DriveSmart AI. Ask me anything about traffic laws. I’ll respond with clear guidance based on the indexed jurisdictions.",
            "metadata": {"sources_count": 0, "response_time": 0.0, "jurisdiction": "All"}
        }
    ]
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "thinking_rendered" not in st.session_state:
    st.session_state.thinking_rendered = False

# -------------------- LEGACY HTML CLEANUP --------------------
TAG_RE = re.compile(r"<[^>]+>")

def sanitize_text(text: str) -> str:
    """Always-safe cleanup for any legacy HTML or accidental UI dumps."""
    if not text:
        return ""

    # Normalize line breaks from old HTML
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # Remove ANY html tags
    text = TAG_RE.sub("", text)

    # Unescape entities (&lt; etc)
    text = html.unescape(text)

    # Clean excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# -------------------- TABLE JSON EXTRACTOR --------------------
TABLE_BLOCK_RE = re.compile(
    r"(?:TABLE__JSON|TABLE_JSON)\s*:\s*([\s\S]*?)(?=\n\s*\n|KEY TAKEAWAYS:|🔑|$)",
    re.IGNORECASE
)

def extract_table_obj_and_clean_text(raw: str):
    """
    Returns (table_obj_or_None, cleaned_display_text)
    Finds TABLE_JSON: { ... } block and removes it from text.
    """
    if not raw:
        return None, raw

    m = TABLE_BLOCK_RE.search(raw)
    if not m:
        return None, raw

    block = (m.group(1) or "").strip()

    # Try to locate JSON object boundaries safely
    start = block.find("{")
    end = block.rfind("}")
    if start == -1 or end == -1 or end <= start:
        display_text = TABLE_BLOCK_RE.sub("", raw).strip()
        return None, display_text

    json_str = block[start:end + 1]

    try:
        table_obj = json.loads(json_str)
    except Exception:
        display_text = TABLE_BLOCK_RE.sub("", raw).strip()
        return None, display_text

    # Remove the entire TABLE_JSON block from visible text
    display_text = TABLE_BLOCK_RE.sub("", raw).strip()

    return table_obj, display_text


def build_comparison_text(table_obj: dict) -> str:
    j1 = str(table_obj.get("jurisdiction_1", "Jurisdiction 1")).strip()
    j2 = str(table_obj.get("jurisdiction_2", "Jurisdiction 2")).strip()
    rows = table_obj.get("rows", []) or []

    lines = []
    lines.append(f"Comparison: {j1} vs {j2}")

    for r in rows:
        aspect = str(r.get("aspect", "")).strip()
        v1 = str(r.get("j1", "")).strip()
        v2 = str(r.get("j2", "")).strip()

        if not aspect:
            continue

        lines.append("")
        lines.append(f"{aspect}")
        if v1:
            lines.append(f"- {j1}: {v1}")
        if v2:
            lines.append(f"- {j2}: {v2}")

    return "\n".join(lines).strip()
def normalize_messages_once():
    cleaned = []
    for m in st.session_state.messages:
        raw = m.get("content", "") or ""
        m["content"] = sanitize_text(raw)
        cleaned.append(m)
    st.session_state.messages = cleaned


# ✅ Run migration only once per session
if "messages_migrated_v3" not in st.session_state:
    normalize_messages_once()
    st.session_state.messages_migrated_v3 = True
# -------------------- DATABASE MANAGER --------------------
class DatabaseManager:
    def __init__(self):
        # ---- Chroma Cloud collection ----
        self.db_connected = False
        self.traffic_collection = None

        try:
            from chromadb import HttpClient
            from chromadb.utils import embedding_functions

            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-3-small"
            )

            self.chroma_client = HttpClient(
                host="api.trychroma.com",
                port=443,
                ssl=True,
                headers={
                    "Authorization": f"Bearer {os.getenv('CHROMA_API_KEY')}",
                    "X-Chroma-Token": os.getenv('CHROMA_API_KEY')
                },
                tenant=os.getenv('CHROMA_TENANT'),
                database=os.getenv('CHROMA_DB')
            )

            self.traffic_collection = self.chroma_client.get_collection(
                "traffic_laws",
                embedding_function=self.embedding_function
            )
            self.db_connected = True
        except Exception:
            self.db_connected = False

        # ---- SQLite for analytics ----
        # Use a writable location for the SQLite file (Streamlit Cloud's repo mount can be read-only).
        import tempfile
        db_dir = os.getenv('DRIVESMART_DB_PATH') or tempfile.gettempdir()
        try:
            os.makedirs(db_dir, exist_ok=True)
        except Exception:
            # If path is a file path or cannot be created, fall back to tempfile.gettempdir()
            db_dir = tempfile.gettempdir()
        self.sqlite_path = str(Path(db_dir) / 'drivesmart_analytics.db')

        # Connect with a timeout to reduce "database is locked" OperationalErrors
        try:
            self.sqlite_conn = sqlite3.connect(self.sqlite_path, check_same_thread=False, timeout=30)
            try:
                # Improve concurrency characteristics
                cur = self.sqlite_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                cur.execute("PRAGMA temp_store=MEMORY;")
            except Exception:
                # If pragmas fail, continue — not fatal
                pass
            # Try to initialize tables; if it fails, we'll fallback to an in-memory DB
            try:
                self._init_sqlite_tables()
            except sqlite3.OperationalError as e:
                print(f"[DatabaseManager] _init_sqlite_tables OperationalError: {e}")
                # Fallback to in-memory DB to keep app running
                try:
                    self.sqlite_conn = sqlite3.connect(':memory:', check_same_thread=False, timeout=30)
                    self._init_sqlite_tables()
                    self.sqlite_path = ':memory:'
                except Exception as e2:
                    print(f"[DatabaseManager] Failed to initialize in-memory DB: {e2}")
                    # Disable analytics by setting sqlite_conn to None
                    self.sqlite_conn = None
        except Exception as e:
            print(f"[DatabaseManager] Could not open SQLite DB at {self.sqlite_path}: {e}")
            # Fallback to in-memory DB if possible
            try:
                self.sqlite_conn = sqlite3.connect(':memory:', check_same_thread=False, timeout=30)
                self._init_sqlite_tables()
                self.sqlite_path = ':memory:'
            except Exception as e2:
                print(f"[DatabaseManager] Could not open in-memory SQLite DB: {e2}")
                self.sqlite_conn = None

    def _init_sqlite_tables(self):
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT,
                jurisdiction TEXT,
                analysis_type TEXT,
                response_time REAL,
                sources_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.sqlite_conn.commit()

    def save_query(self, query_data):
        if not self.sqlite_conn:
            # Analytics disabled / DB not available
            return None

        cursor = self.sqlite_conn.cursor()
        sql = '''
                INSERT INTO query_history 
                (query, response, jurisdiction, analysis_type, response_time, sources_count)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
        params = (
            query_data.get('query', ''),
            (query_data.get('response', '') or '')[:1000],
            query_data.get('jurisdiction', 'All'),
            query_data.get('analysis_type', 'general'),
            float(query_data.get('response_time', 0.0)),
            int(query_data.get('sources_count', 0))
        )

        # Retry loop to handle transient "database is locked" errors
        for attempt in range(3):
            try:
                cursor.execute(sql, params)
                self.sqlite_conn.commit()
                return cursor.lastrowid
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                try:
                    print(f"[DatabaseManager] OperationalError on save_query attempt {attempt+1}: {e}")
                except Exception:
                    pass
                # If it's a lock, wait a bit and retry
                if 'locked' in msg or 'database is locked' in msg:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                # Otherwise, don't retry
                return None
            except Exception as e:
                try:
                    print(f"[DatabaseManager] Unexpected error saving query: {e}")
                except Exception:
                    pass
                return None

        # If we exhausted retries, surface a concise log and return
        try:
            print("[DatabaseManager] save_query failed after retries")
        except Exception:
            pass
        return None

    def get_stats(self):
        cursor = self.sqlite_conn.cursor()

        today_stats = cursor.execute('''
            SELECT COUNT(*) as queries_today, AVG(response_time) as avg_response_time
            FROM query_history
            WHERE DATE(timestamp) = DATE('now', 'localtime')
        ''').fetchone()

        total_stats = cursor.execute('''
            SELECT COUNT(*) as total_queries
            FROM query_history
        ''').fetchone()

        laws_count = 24
        if self.db_connected and self.traffic_collection:
            try:
                laws_count = self.traffic_collection.count()
            except Exception:
                pass

        return {
            'queries_today': today_stats[0] if today_stats else 0,
            'avg_response_time': today_stats[1] if today_stats and today_stats[1] else 2.1,
            'total_queries': total_stats[0] if total_stats else 0,
            'laws_indexed': laws_count
        }

    def get_supported_jurisdictions(self, limit=2000):
        fallback = ["Massachusetts", "California", "New York", "Texas", "Florida"]

        if not self.db_connected or not self.traffic_collection:
            return fallback

        try:
            data = self.traffic_collection.get(include=["metadatas"], limit=limit)
            metas = data.get("metadatas", []) or []

            flattened = []
            for m in metas:
                if isinstance(m, list):
                    flattened.extend(m)
                elif isinstance(m, dict):
                    flattened.append(m)

            jurisdictions = sorted({
                str(m.get("jurisdiction")).strip()
                for m in flattened
                if m and m.get("jurisdiction")
            })

            return jurisdictions if jurisdictions else fallback
        except Exception:
            return fallback

# -------------------- CACHED RESOURCES --------------------
@st.cache_resource
def get_managers():
    db_manager = DatabaseManager()

    if SECURITY_ENABLED:
        input_validator = PromptSecurityValidator()
        output_validator = ResponseValidator()
        behavioral_monitor = BehavioralMonitor()
        return db_manager, input_validator, output_validator, behavioral_monitor

    return db_manager, None, None, None

@st.cache_resource
def get_vectorstore():
    vsm = CloudTrafficLawVectorStore()
    return vsm.get_existing_vectorstore("traffic_laws")


@st.cache_resource
def get_workflow():
    try:
        from SmartDrive.src.refined_prompts import RefinedDriveSmartWorkflow
        from SmartDrive.src.vector_store import CloudTrafficLawVectorStore
    except Exception as e:
        st.error(f"❌ Could not load DriveSmart core modules: {e}")
        return None

    try:
        init_params = list(inspect.signature(RefinedDriveSmartWorkflow.__init__).parameters)

        if len(init_params) <= 1:
            return RefinedDriveSmartWorkflow()

        vsm = CloudTrafficLawVectorStore()
        vectorstore = vsm.get_existing_vectorstore("traffic_laws")
        return RefinedDriveSmartWorkflow(vectorstore)

    except Exception as e:
        st.error(f"❌ Failed to initialize workflow: {e}")
        return None
db_manager, input_validator, output_validator, behavioral_monitor = get_managers()
workflow = get_workflow()
if not workflow:
    st.stop()
supported_states = db_manager.get_supported_jurisdictions()


# -------------------- MODERN CSS --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    :root{
        --accent-red: #BF092F;
        --navy-deep: #132440;
        --blue-steel: #16476A;
        --teal: #3B9797;

        --bg-0: #050607;
        --bg-1: #0A0D12;
        --bg-2: #0D1117;

        --glass-1: rgba(255,255,255,0.06);
        --glass-2: rgba(255,255,255,0.09);
        --border-1: rgba(255,255,255,0.08);

        --text-strong: #F8FAFC;
        --text-mid: rgba(248,250,252,0.78);
        --text-dim: rgba(248,250,252,0.55);

        --shadow-soft: 0 8px 28px rgba(0,0,0,0.35);
        --shadow-pop: 0 10px 30px rgba(0,0,0,0.45);
        --base-font-size: 13px;
    }
/* Import Poppins */
section[data-testid="stSidebar"] div[data-testid="stButton"]:has(button):has(span:contains("Clear Chat History")) button{
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
    box-shadow: none !important;
}
/* Sidebar - make ONLY "Clear Chat History" white with black text */
section[data-testid="stSidebar"] button[aria-label="🗑️ Clear Chat History"],
section[data-testid="stSidebar"] button:has(span:contains("Clear Chat History")) {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
    box-shadow: none !important;
}

/* Hover */
section[data-testid="stSidebar"] button[aria-label="🗑️ Clear Chat History"]:hover {
    background: #f5f5f5 !important;
    color: #000000 !important;
    transform: none !important;
    filter: none !important;
}
/* Apply globally */
html, body, [class*="st-"], .stApp, .stMarkdown, .stTextArea, .stButton, input, textarea, button {
    font-family: "Poppins", system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Reduce global font size for denser layout */
html, body, .stApp, .stMarkdown {
    font-size: var(--base-font-size) !important;
}

/* Ensure your custom header text also uses Poppins */
header[data-testid="stHeader"]::after,
div[data-testid="stToolbar"]::before{
    font-family: "Poppins", system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, sans-serif !important;
}
    .stApp {
        background:
            radial-gradient(circle at 12% 12%, rgba(59,151,151,0.16), transparent 38%),
            radial-gradient(circle at 88% 20%, rgba(22,71,106,0.16), transparent 42%),
            radial-gradient(circle at 40% 90%, rgba(191,9,47,0.10), transparent 45%),
            linear-gradient(135deg, var(--bg-0) 0%, var(--bg-1) 45%, var(--bg-2) 100%) !important;
        color: var(--text-strong);
    }
/* --- Top Streamlit header bar --- */
header[data-testid="stHeader"]{
    background: #000000 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    box-shadow: none !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 9999 !important;
}

/* The tiny top accent/decoration strip */
div[data-testid="stDecoration"]{
    background: #000000 !important;
    height: 0px !important;  /* remove the colored line */
}

/* Optional: make toolbar area also black */
div[data-testid="stToolbar"]{
    background: #000000 !important;
}

/* Hide default Streamlit header actions if they still appear */
.stDeployButton { display: none !important; }
#MainMenu { visibility: hidden !important; }
.st-emotion-cache-ausnhs{
            
            
            }
/* Add custom header text */
header[data-testid="stHeader"]::after{
    content: "Ask DriveSmartAi";
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.3px;

    position: absolute;
                left: 50%;

    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
}
    section[data-testid="stSidebar"]{
        background:
            linear-gradient(180deg, rgba(19,36,64,0.35), rgba(5,6,7,0.9)) !important;
        border-right: 1px solid var(--border-1);
    }
section[data-testid="stSidebar"]{
    background: #050607 !important; /* same as --bg-0 */
    border-right: 1px solid var(--border-1);
}

    .chat-header {
        background: linear-gradient(
            135deg,
            rgba(22,71,106,0.18),
            rgba(59,151,151,0.12),
            rgba(19,36,64,0.18)
        );
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 1.6rem 1.4rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-1);
        text-align: center;
        box-shadow: var(--shadow-soft);
    }

    .chat-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: var(--text-strong);
        margin-bottom: 0.25rem;
    }

    .chat-subtitle {
        color: var(--text-mid);
        font-size: 1rem;
    }

    .supported-strip {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        padding: 0.85rem 1rem;
        margin: 0 0 1.1rem 0;
        border-radius: 14px;
        background: var(--glass-1);
        border: 1px solid var(--border-1);
        box-shadow: var(--shadow-soft);
    }

    .supported-label {
        font-weight: 700;
        color: var(--text-mid);
        margin-right: 0.25rem;
    }

    .state-pill {
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 1.05rem;
        color: #dff7f7;
        background: rgba(59,151,151,0.18);
        border: 1px solid rgba(59,151,151,0.35);
    }

    .state-more{
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        color: var(--text-mid);
        background: rgba(255,255,255,0.06);
        border: 1px solid var(--border-1);
    }
/* ------------------ CHAT ROWS (NO WRAPPER) ------------------ */
.message-row{
    display: flex;
    width: 100%;
    margin: 0.9rem 0;
}

.message-row.user{
    justify-content: flex-end;
}

.message-row.ai{
    justify-content: flex-start;
}

/* ------------------ USER BUBBLE ------------------ */
.user-message{
    background: linear-gradient(135deg, var(--teal), var(--blue-steel)) !important;
    color: black;
    padding: 1rem 1.2rem;
    border-radius: 16px 16px 4px 16px;
    max-width: 70%;
    box-shadow: 0 6px 18px rgba(191,9,47,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}

/* ------------------ AI BUBBLE ------------------ */
.ai-message{
    background: linear-gradient(
        135deg,
        rgba(22,71,106,0.22),
        rgba(19,36,64,0.25)
    );
    color: var(--text-strong);
    padding: 1.1rem 1.2rem;
    border-radius: 16px 16px 16px 4px;
    max-width: 82%;
    border-left: 4px solid var(--teal);
    box-shadow: 0 6px 16px rgba(0,0,0,0.25);
}

/* Meta stays same */
.message-meta {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 0.55rem;
    display: flex;
    gap: 0.9rem;
    align-items: center;
}
    # .chat-container {
    #     background: linear-gradient(
    #         180deg,
    #         rgba(255,255,255,0.05),
    #         rgba(255,255,255,0.03)
    #     );
    #     backdrop-filter: blur(10px);
    #     border-radius: 18px;
    #     padding: 1.25rem 1.25rem 0.6rem 1.25rem;
    #     border: 1px solid var(--border-1);
    #     box-shadow: var(--shadow-pop);

    #     height: calc(100vh - 460px);
    #     min-height: 280px;
    #     overflow-y: auto;
    #     margin-bottom: 1rem;
    # }

    .user-message,
    .ai-message {
        font-size: 1rem;
        line-height: 1.6;
    }

    .user-message {
        background: linear-gradient(135deg, var(--teal), var(--blue-steel)) !important;
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.9rem 0 0.9rem auto;
        max-width: 70%;
        float: right;
        clear: both;
        box-shadow: 0 6px 18px rgba(191,9,47,0.35);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .ai-message {
        background: linear-gradient(
            135deg,
            rgba(22,71,106,0.22),
            rgba(19,36,64,0.25)
        );
        color: var(--text-strong);
        padding: 1.1rem 1.2rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.9rem 0;
        max-width: 82%;
        border-left: 4px solid var(--teal);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }

    .message-meta {
        font-size: 0.8rem;
        color: var(--text-dim);
        margin-top: 0.55rem;
        display: flex;
        gap: 0.9rem;
        align-items: center;
    }

    .message-clear {
        clear: both;
        content: "";
        display: table;
    }

    .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid var(--border-1) !important;
        border-radius: 10px !important;
        color: black !important;
        font-size: 1rem !important;
        padding: 0.95rem !important;
        caret-color: var(--teal) !important;
    }

    .stTextArea textarea::placeholder{
        color: #6B7280 !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(59,151,151,0.6) !important;
        box-shadow: 0 0 0 3px rgba(59,151,151,0.18) !important;
    }

    .stButton button {
        background: linear-gradient(135deg, var(--accent-red), #8c0722);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        padding: 0.82rem 1.15rem !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25) !important;
    }

    .stButton button:hover {
        transform: translateY(-1px) !important;
        filter: brightness(1.08) !important;
        box-shadow: 0 10px 26px rgba(0,0,0,0.35) !important;
    }

    .stat-card {
        background: var(--glass-1);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        border: 1px solid var(--border-1);
        text-align: center;
        box-shadow: var(--shadow-soft);
    }

    .stat-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #dff7f7;
    }

    .stat-label {
        color: var(--text-dim);
        font-size: 1rem;
        margin-top: 0.2rem;
    }

    .security-badge {
        display: inline-block;
        background: rgba(59,151,151,0.16);
        color: #dff7f7;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid rgba(59,151,151,0.35);
        margin-bottom: 0.5rem;
    }

    .security-badge.warning {
        background: rgba(191,9,47,0.12);
        color: #ffb3c2;
        border-color: rgba(191,9,47,0.35);
    }
/* -------- Sticky Bottom Composer -------- */

/* Give page extra room so content doesn't hide behind fixed composer */
/* -------- Sticky Bottom Composer (robust) -------- */

/* Give extra room so content doesn't hide behind fixed composer */
/* -------- Sticky Bottom Composer (single source of truth) -------- */

/* Give extra room so content doesn't hide behind fixed composer */
div[data-testid="stAppViewContainer"] .main .block-container{
    padding-bottom: 10px !important;
}
/* Space so content doesn't hide behind fixed composer */
div[data-testid="stAppViewContainer"] .main .block-container{
    padding-bottom: 10px !important;
}

/* Make ONLY the immediate block after the anchor fixed */
#composer-anchor + div{
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;

    background: linear-gradient(
        180deg,
        rgba(5,6,7,0.0),
        rgba(5,6,7,0.88),
        #050607
    ) !important;

    backdrop-filter: blur(12px);
    border-top: 1px solid rgba(255,255,255,0.08) !important;

    padding: 0.9rem 1.2rem 0.7rem 1.2rem !important;
    z-index: 9999 !important;
}

.composer-footer{
    text-align: center;
    color: rgba(255,255,255,0.5);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    line-height: 1.3;
}
/* Fix ONLY the block that contains our marker */

section[data-testid="stSidebar"] button[aria-label="🗑️ Clear Chat History"] {
    background: #fff !important;
    color: #000 !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
}

/* Target by key hint (Streamlit often includes key in attributes/classes) */

/* Footer style inside composer */


/* The block right after this anchor becomes the fixed bottom bar */


/* Make sure textarea + button don't stretch weirdly inside fixed container */

    hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.08) !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        min-height: 52px !important;
        padding: 0.9rem 1.1rem !important;
        font-size: 1.05rem !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR --------------------
with st.sidebar:
 
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] button[aria-label="🗑️ Clear Chat History"]{
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid rgba(0,0,0,0.15) !important;
        box-shadow: none !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] button[aria-label="🗑️ Clear Chat History"]:hover{
        background: #f5f5f5 !important;
        color: #000000 !important;
        transform: none !important;
        filter: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="chat-header">DASHBOARD', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    stats = db_manager.get_stats()

    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{stats['queries_today']}</div>
        <div class="stat-label">Queries Today</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{stats['avg_response_time']:.1f}s</div>
        <div class="stat-label">Avg Response Time</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{stats['laws_indexed']}</div>
        <div class="stat-label">Laws Indexed</div>
    </div>
    """, unsafe_allow_html=True)

    if SECURITY_ENABLED:
        st.markdown('<div class="security-badge">🛡️ Security Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="security-badge warning">⚠️ Security Disabled</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h3 style="color: white;">💡 Quick Prompts</h3>', unsafe_allow_html=True)
    # -------------------- CLEAR INPUT (MUST RUN BEFORE text_area) --------------------
    
    quick_prompts = [
        "What is the basic speed law in California?",
        "Can I turn right on red in New York?",
        "What happens if I run a red light?",
        "DUI laws in Massachusetts",
        "Using a phone while driving penalties"
    ]

    for prompt in quick_prompts:
        if st.button(prompt, key=f"quick_{prompt}", use_container_width=True):
            st.session_state.quick_prompt = prompt

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", key="clear_chat", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.messages_migrated_v3 = True  # keep as migrated
        st.rerun()

# -------------------- HEADER --------------------
st.markdown("""
<div class="chat-header">
    <div class="chat-title">🚗 DriveSmart AI</div>
    <div class="chat-subtitle">Your AI-Powered Traffic Law Assistant</div>
</div>
""", unsafe_allow_html=True)
# -------------------- SUPPORTED STATES STRIP --------------------
def render_state_pills(items):
    items = items or []
    pills = "".join(f'<span class="state-pill">{s}</span>' for s in items[:12])
    more = f'<span class="state-more">+{len(items)-12} more</span>' if len(items) > 12 else ""
    return pills + more

st.markdown(
    f"""
    <div class="supported-strip">
        <span class="supported-label">Supported States:</span>
        {render_state_pills(supported_states)}
    </div>
    """,
    unsafe_allow_html=True
)

def get_meta(message: dict):
    # Support BOTH:
    # 1) new shape: message["metadata"] = {...}
    # 2) old shape: message["sources_count"] etc at top-level
            md = message.get("metadata")
            if isinstance(md, dict):
               return md

            return {
                "sources_count": message.get("sources_count", 0),
                "response_time": message.get("response_time", 0.0),
                "jurisdiction": message.get("jurisdiction", "All"),
                "out_of_scope": message.get("out_of_scope", False),
                "is_thinking": message.get("is_thinking", False),
            }
def assistant_msg(content: str, sources_count=0, response_time=0.0, jurisdiction="All",
                  out_of_scope=False, is_thinking=False):
    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "sources_count": int(sources_count),
            "response_time": float(response_time),
            "jurisdiction": jurisdiction,
            "out_of_scope": bool(out_of_scope),
            "is_thinking": bool(is_thinking),
        }
    }
# -------------------- CHAT WINDOW --------------------
def render_chat(messages):
    for message in messages:
        role = message.get("role")
        raw_content = message.get("content", "")
        # metadata = message.get("metadata", {})
        metadata = get_meta(message)

        clean = sanitize_text(raw_content)
        content = html.escape(clean).replace("\n", "<br/>")

        if role == "user":
            st.markdown(
                f"""
                <div class="message-row user">
                    <div class="user-message">
                        <strong>You</strong><br/>
                        {content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            sources = metadata.get("sources_count", 0)
            response_time = metadata.get("response_time", 0.0)
            jurisdiction = metadata.get("jurisdiction", "All")

            st.markdown(
                f"""
                <div class="message-row ai">
                    <div class="ai-message">
                        <strong>🤖 DriveSmart AI</strong><br/>
                        {content}
                        <div class="message-meta">
                            <span>⏱️ {response_time:.2f}s</span>
                            <span>📚 {sources} sources</span>
                            <span>📍 {jurisdiction}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
render_chat(st.session_state.messages)

# ✅ Intro image only until the first user question
has_user_message = any(m.get("role") == "user" for m in st.session_state.messages)

if not has_user_message:
    try:
        # Use absolute path from the module root so Streamlit Cloud finds the file
        img_path = ROOT / "intro.png"
        if img_path.exists():
            try:
                st.image(str(img_path), width=500)
            except Exception:
                # If Streamlit fails to render the image for any reason, continue
                pass
        else:
            # Intro image missing in deployment — skip gracefully
            pass
    except Exception:
        # Defensive: do not allow any image-handling error to crash the app
        pass

# -------------------- QUICK PROMPT PREFILL (BEFORE COMPOSER) --------------------
# This ensures the textarea shows the selected prompt on rerun
# -------------------- QUICK PROMPT PREFILL (BEFORE COMPOSER) --------------------
# -------------------- QUICK PROMPT PREFILL (BEFORE COMPOSER) --------------------
# Initialize default value for the text area
default_value = ""
if "quick_prompt" in st.session_state:
    default_value = st.session_state.quick_prompt
    st.session_state["auto_send"] = True
    del st.session_state.quick_prompt


# -------------------- BOTTOM COMPOSER (INPUT + FOOTER) --------------------
# ... header + supported strip + render_chat above ...

# Anchor right before composer
# Anchor right before composer
with st.form("composer_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_area(
            "Ask a question",
            value=default_value,  # ✅ Use value instead of key binding
            placeholder="Type your traffic law question here...",
            height=90,
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send_button = st.form_submit_button("Ask Now", use_container_width=True, type="primary")

    st.markdown("""
    <div class="composer-footer">
        🚗 DriveSmart AI © 2025 | Powered by LangChain, ChromaDB & OpenAI<br/>
        Northeastern University - INFO 7375
    </div>
    """, unsafe_allow_html=True)
# -------------------- AUTO-SEND FOR QUICK PROMPTS --------------------
if st.session_state.pop("auto_send", False):
    send_button = True


# -------------------- PROCESS MESSAGE --------------------
# -------------------- PROCESS MESSAGE (PHASE 1: capture + clear) --------------------
if send_button and user_input and not st.session_state.processing:
    clean_user = sanitize_text(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": clean_user
    })

    st.session_state.pending_query = clean_user
    st.session_state.processing = True
    st.session_state.thinking_rendered = False


    # 2) stash pending query for async-like UX
    st.session_state.pending_query = clean_user
    st.session_state.processing = True
    st.session_state.thinking_rendered = False

   

    st.rerun()


# -------------------- PROCESS MESSAGE (PHASE 2/3: thinking -> answer) --------------------
if st.session_state.processing and st.session_state.pending_query:
    pending = st.session_state.pending_query

    # Phase 2: render a visible thinking bubble
    if not st.session_state.thinking_rendered:
        st.session_state.messages[-1] = assistant_msg(
            answer_text,
            sources_count=sources_count,
            response_time=response_time,
            jurisdiction=jur,
            out_of_scope=False,
            is_thinking=False
        )
        st.session_state.thinking_rendered = True
        st.rerun()

    # Phase 3: compute and replace the thinking bubble
    start_time = time.time()

    # ---------------- SECURITY VALIDATION ----------------
    if SECURITY_ENABLED and input_validator and behavioral_monitor:
        validation = input_validator.validate_input(pending)
        behavioral = behavioral_monitor.analyze_session(
            st.session_state.session_id,
            pending,
            validation
        )

        is_safe = bool(validation.get("is_safe", True))
        risk_level = validation.get("risk_level", "unknown")
        action = behavioral.get("action", "ALLOW")

        if (not is_safe) or action in ["RATE_LIMIT", "BLOCK"]:
            block_text = f"🚫 **Security Alert:** This query was blocked. Risk level: **{risk_level}**."

            # replace last thinking msg
            if st.session_state.messages and st.session_state.messages[-1].get("metadata", {}).get("is_thinking"):
                st.session_state.messages[-1] = assistant_msg(
                    oos_text,
                    sources_count=0,
                    response_time=0.0,
                    jurisdiction="N/A",
                    out_of_scope=True,
                     is_thinking=False
                )
                
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": block_text,
                    "metadata": {"sources_count": 0, "response_time": 0.0, "jurisdiction": "N/A"}
                })

            db_manager.save_query({
                "query": pending,
                "response": block_text,
                "jurisdiction": "N/A",
                "analysis_type": "Security Block",
                "response_time": 0.0,
                "sources_count": 0
            })

            st.session_state.processing = False
            st.session_state.pending_query = None
            st.session_state.thinking_rendered = False
            st.rerun()

    # ---------------- SCOPE GUARD ----------------
    if not is_traffic_related(pending):
        oos_text = build_out_of_scope_answer(supported_states)

        if st.session_state.messages and st.session_state.messages[-1].get("metadata", {}).get("is_thinking"):
            st.session_state.messages[-1] = {
                "role": "assistant",
                "content": oos_text,
                "metadata": {
                    "sources_count": 0,
                    "response_time": 0.0,
                    "jurisdiction": "N/A",
                    "out_of_scope": True,
                    "is_thinking": False
                }
            }
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": oos_text,
                "metadata": {"sources_count": 0, "response_time": 0.0, "jurisdiction": "N/A"}
            })

        db_manager.save_query({
            "query": pending,
            "response": oos_text,
            "jurisdiction": "N/A",
            "analysis_type": "out_of_scope",
            "response_time": 0.0,
            "sources_count": 0
        })

        st.session_state.processing = False
        st.session_state.pending_query = None
        st.session_state.thinking_rendered = False
        st.rerun()

    # ---------------- PROMPT TYPE DETECTION ----------------
    q_lower = pending.lower()
    if "compare" in q_lower or "difference" in q_lower:
        prompt_type_key = "comparative"
    elif "i was" in q_lower or "scenario" in q_lower:
        prompt_type_key = "scenario"
    else:
        prompt_type_key = "general"

    # ---------------- RUN WORKFLOW ----------------
    result = workflow.query(pending, prompt_type_key)
    response_time = time.time() - start_time

    jur = result.get("detected_jurisdiction", "All")
    if isinstance(jur, list):
        jur = ", ".join(jur)

    raw_answer = (result.get("answer") or "").strip()
    table_obj, display_text = extract_table_obj_and_clean_text(raw_answer)
    answer_text = sanitize_text(display_text)

    if table_obj:
        comparison_text = build_comparison_text(table_obj)
        answer_text = f"{answer_text}\n\n{comparison_text}".strip()

    sources_count = len(result.get("sources", []) or [])

    # replace last thinking msg with final answer
    if st.session_state.messages and st.session_state.messages[-1].get("metadata", {}).get("is_thinking"):
        st.session_state.messages[-1] = {
            "role": "assistant",
            "content": answer_text,
            "metadata": {
                "sources_count": sources_count,
                "response_time": response_time,
                "jurisdiction": jur,
                "out_of_scope": False,
                "is_thinking": False
            }
        }
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "metadata": {
                "sources_count": sources_count,
                "response_time": response_time,
                "jurisdiction": jur,
                "out_of_scope": False,
                "is_thinking": False
            }
        })

    db_manager.save_query({
        "query": pending,
        "response": answer_text,
        "jurisdiction": jur,
        "analysis_type": prompt_type_key,
        "response_time": response_time,
        "sources_count": sources_count
    })

    # Clear the composer input so the textarea resets after an answer
    try:
        st.session_state["user_input"] = ""
    except Exception:
        # If session state isn't available for any reason, continue gracefully
        pass

    st.session_state.processing = False
    st.session_state.pending_query = None
    st.session_state.thinking_rendered = False
    st.rerun()