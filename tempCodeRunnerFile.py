# Check for XGBoost availability
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Check for SHAP availability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Session keys
SESSION_RAW = "__uploaded_df__"
SESSION_CLEAN = "__cleaned_df__"
