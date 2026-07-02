import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Project Metadata
PROJECT_NAME = "EdgeExplain"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = (
    "Completely offline Edge AI platform for dataset analysis, data quality "
    "evaluation, feature relationship explanation, and preprocessing recommendations."
)

# Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

# Ensure directories exist
for directory in [DATA_DIR, REPORTS_DIR, LOGS_DIR, ASSETS_DIR, KNOWLEDGE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Log Config
LOG_FILE_PATH = os.path.join(LOGS_DIR, "edgeexplain.log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Supported Formats
SUPPORTED_FORMATS = {
    "csv": [".csv"],
    "excel": [".xlsx", ".xls"],
    "json": [".json"],
    "parquet": [".parquet", ".pq"]
}

# Max upload limit (in bytes) - default 100MB
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  

# Styling CSS Theme Customizations (Hex colors for Streamlit markup injection)
THEME_COLORS = {
    "primary": "#1A56DB",       # Royal Blue
    "secondary": "#7E3AF2",     # Purple
    "background": "#0F172A",    # Dark Slate
    "card_bg": "#1E293B",       # Slate 800
    "text": "#F8FAFC",          # Slate 50
    "text_muted": "#94A3B8",    # Slate 400
    "border": "#334155",        # Slate 700
    "success": "#10B981",       # Emerald Green
    "warning": "#F59E0B",       # Amber Orange
    "danger": "#EF4444"         # Red
}

# Domain detection vocabulary (used by Intelligence Engine)
DOMAIN_VOCABULARIES = {
    "Education": [
        "student", "class", "course", "grade", "cgpa", "attendance", "semester", 
        "exam", "teacher", "gpa", "school", "university", "faculty", "enrollment"
    ],
    "Healthcare": [
        "patient", "blood", "heart", "disease", "diagnosis", "symptoms", "age", 
        "bmi", "treatment", "clinical", "hospital", "doctor", "cholesterol", "diabetes"
    ],
    "Finance": [
        "transaction", "amount", "balance", "credit", "debit", "loan", "interest", 
        "stock", "rate", "revenue", "price", "asset", "liability", "expense", "portfolio"
    ],
    "Retail": [
        "product", "sales", "customer", "order", "discount", "invoice", "store", 
        "inventory", "sku", "purchase", "retailer", "cart", "transaction_id"
    ],
    "IoT": [
        "sensor", "temp", "temperature", "humidity", "voltage", "device", "telemetry", 
        "signal", "reading", "vibration", "sensor_id", "frequency", "current"
    ],
    "Manufacturing": [
        "machine", "cycle", "defect", "assembly", "downtime", "batch", "yield", 
        "production", "operator", "equipment", "maintenance", "throughput"
    ],
    "Government": [
        "census", "population", "district", "municipality", "tax", "vote", "citizen", 
        "welfare", "budget", "policy", "election", "county", "demographics"
    ],
    "Transportation": [
        "route", "speed", "vehicle", "driver", "trip", "distance", "passenger", 
        "flight", "delay", "gps", "destination", "origin", "fare"
    ],
    "Survey": [
        "respondent", "opinion", "rating", "feedback", "satisfaction", "demographic", 
        "gender", "agree", "disagree", "response", "survey_id"
    ],
    "Social Media": [
        "tweet", "post", "like", "share", "follower", "comment", "engagement", 
        "hashtag", "views", "user_id", "retweet", "subscriber"
    ]
}

# Configurable analytical threshold defaults
DEFAULT_OUTLIER_THRESHOLD = 1.5
DEFAULT_CORRELATION_THRESHOLD = 0.3
DEFAULT_THEME = "Dark Slate"
