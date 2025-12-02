# data_upload.py
"""
Data Upload page module for AutoAnalyst AI (auto-saves uploaded DataFrame to session).

- Upload CSV / Excel
- Generate a sample dataset
- Connect to SQL databases (if you have that version)
- Automatically saves read DataFrame into st.session_state["__uploaded_df__"]
- Keeps metadata in st.session_state["__uploaded_meta__"]
"""

import streamlit as st
import pandas as pd
import io
from typing import Tuple, Optional

SESSION_KEY_UPLOADED_DF = "__uploaded_df__"   # session key used across app
SESSION_KEY_UPLOADED_META = "__uploaded_meta__"

# ----------------------------
# Helpers
# ----------------------------
def _read_file(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Try to read uploaded file (csv / xlsx). Returns (df, error) where error is None on success.
    """
    fname = file.name.lower()
    try:
        if fname.endswith(".csv"):
            # try usual encodings and fallback to latin1
            try:
                df = pd.read_csv(file)
            except Exception:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin1")
            return df, None
        elif fname.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
            return df, None
        else:
            return None, f"Unsupported file type: {fname.split('.')[-1]}"
    except Exception as e:
        return None, str(e)

def _generate_sample_df():
    """Return a small demo DataFrame (for the sample CSV generator)."""
    df = pd.DataFrame({
        "id": range(1, 11),
        "name": ["Alice","Bob","Charlie","David","Eva","Fay","Gina","Hugo","Isha","Jin"],
        "age": [25, 34, 28, 45, 30, 22, 36, 41, 27, 33],
        "salary": [50000, 72000, 61000, 98000, 54000, 45000, 67000, 88000, 52000, 60000],
        "department": ["sales","hr","eng","eng","marketing","sales","eng","hr","marketing","sales"],
        "joined": pd.date_range("2020-01-01", periods=10, freq="90D").astype(str)
    })
    return df

# ----------------------------
# Page renderer
# ----------------------------
def render_data_upload():
    """
    Render the Data Upload page.
    - Uses st.file_uploader
    - On successful read, automatically store DataFrame in st.session_state[SESSION_KEY_UPLOADED_DF]
    - Shows preview, dtypes, and basic actions (clear, show session dataset)
    """
    st.header("Data Upload")
    st.write("Upload a CSV or Excel file. Uploaded dataset is stored in the session for use by cleaning / EDA pages.")

    col_upload, col_actions = st.columns([3, 1], gap="small")
    with col_upload:
        uploaded = st.file_uploader("Choose a file (CSV / Excel)", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

        st.markdown("**Or** generate a small sample dataset for demo:")
        if st.button("Generate sample dataset (10 rows)"):
            df_sample = _generate_sample_df()
            st.session_state[SESSION_KEY_UPLOADED_DF] = df_sample
            st.session_state[SESSION_KEY_UPLOADED_META] = {"name": "sample_dataset", "rows": df_sample.shape[0], "cols": df_sample.shape[1], "source": "sample"}
            st.success("Sample dataset generated and stored in session.")

    with col_actions:
        st.markdown("**Session actions**")
        if st.button("Clear uploaded dataset"):
            if SESSION_KEY_UPLOADED_DF in st.session_state:
                del st.session_state[SESSION_KEY_UPLOADED_DF]
            if SESSION_KEY_UPLOADED_META in st.session_state:
                del st.session_state[SESSION_KEY_UPLOADED_META]
            st.info("Session dataset cleared.")
        if st.button("Show uploaded session dataset"):
            if SESSION_KEY_UPLOADED_DF in st.session_state:
                df = st.session_state[SESSION_KEY_UPLOADED_DF]
                st.write(f"Session dataset: {st.session_state.get(SESSION_KEY_UPLOADED_META, {}).get('name', 'uploaded (session)')}")
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
                st.dataframe(df.head(50))
            else:
                st.warning("No dataset in session. Upload or generate a sample first.")

    # if user uploaded a file, try to read it and auto-save
    if uploaded is not None:
        with st.spinner("Reading file..."):
            df, err = _read_file(uploaded)
        if err:
            st.error(f"Failed to read file: {err}")
            return

        # successful read
        st.success(f"Loaded `{uploaded.name}` — {df.shape[0]} rows × {df.shape[1]} columns")

        # show basic preview & dtypes
        st.markdown("**Preview (first 50 rows)**")
        st.dataframe(df.head(50))

        st.markdown("**Columns & types**")
        dtypes = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.dataframe(dtypes)

        # AUTO-SAVE: store DF into session immediately so other pages can access it
        meta = {"name": uploaded.name, "rows": df.shape[0], "cols": df.shape[1], "source": "file_upload"}
        st.session_state[SESSION_KEY_UPLOADED_DF] = df
        st.session_state[SESSION_KEY_UPLOADED_META] = meta
        st.info("Uploaded dataset has been saved to session — you can navigate to Data Cleaning or EDA now.")

        # actions: download or view missing values
        c1, c2 = st.columns([1,1], gap="small")
        with c1:
            to_download = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", to_download, file_name=f"{uploaded.name.rsplit('.',1)[0]}_copy.csv", mime="text/csv")
        with c2:
            if st.button("Show missing values"):
                mv = df.isna().sum()
                if (mv>0).any():
                    st.markdown("**Missing values per column**")
                    st.dataframe(mv[mv > 0].sort_values(ascending=False).to_frame("missing_count"))
                else:
                    st.info("No missing values detected in this dataset.")

    else:
        # No uploaded file; show a gentle hint / preview of sample if present in session
        if SESSION_KEY_UPLOADED_DF in st.session_state:
            df = st.session_state[SESSION_KEY_UPLOADED_DF]
            meta = st.session_state.get(SESSION_KEY_UPLOADED_META, {})
            st.info(f"Using session dataset: {meta.get('name','(session)')} — {df.shape[0]} rows × {df.shape[1]} cols")
            st.dataframe(df.head(20))
        else:
            st.info("No dataset loaded. Upload a CSV/Excel file, or press 'Generate sample dataset' to try the app.")

    # Return nothing; page communicates via st.session_state
    return
