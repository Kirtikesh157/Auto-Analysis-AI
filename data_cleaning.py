# data_cleaning.py
"""
Simple & safe Data Cleaning page for AutoAnalyst AI.

Features (minimal but accurate):
- Inspect missingness summary
- Drop columns with missing% >= threshold (slider)
- Drop rows:
    * drop if any missing (global), or
    * drop if missing in a selected set of columns
- Preview before/after, show audit log
- Save cleaned DataFrame to st.session_state["__cleaned_df__"]
- Download cleaned CSV and rollback cleaned dataset

No model-based imputation included (keeps behavior simple & reproducible).
"""

import streamlit as st
import pandas as pd

SESSION_RAW = "__uploaded_df__"
SESSION_CLEAN = "__cleaned_df__"
SESSION_CLEAN_META = "__cleaned_meta__"

def render_data_cleaning():
    st.header("Data Cleaning — Simple & Safe")

    if SESSION_RAW not in st.session_state:
        st.warning("No dataset loaded. Go to Data Upload and load a dataset first.")
        return

    raw_df = st.session_state[SESSION_RAW].copy()
    n_rows, n_cols = raw_df.shape
    st.markdown(f"**Loaded dataset:** {n_rows} rows × {n_cols} columns")

    # --- missingness summary
    st.subheader("Missingness summary")
    miss = raw_df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / len(raw_df)).round(4)
    miss_df = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
    st.dataframe(miss_df.style.format({"missing_pct": "{:.2%}"}))

    st.markdown("---")

    # --- options
    st.subheader("Dropping options (safe defaults)")

    col1, col2 = st.columns(2)
    with col1:
        drop_cols_thresh = st.slider(
            "Drop columns with missing percentage ≥",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f"
        )
        drop_cols = st.checkbox("Enable drop columns above threshold", value=True)
    with col2:
        row_drop_mode = st.selectbox(
            "Row drop option",
            options=[
                "Don't drop rows",
                "Drop rows with any missing value",
                "Drop rows with missing in selected columns"
            ],
            index=0
        )

    cols_to_check = []
    if row_drop_mode == "Drop rows with missing in selected columns":
        # let user pick columns (default: top 5 missing)
        top_missing = miss[miss > 0].index.tolist()[:5]
        cols_to_check = st.multiselect("Select columns to consider for row-drop", options=raw_df.columns.tolist(), default=top_missing)

    st.markdown("---")

    # --- Preview area: compute what WOULD be dropped
    st.subheader("Preview changes (no writes yet)")

    # Determine columns to drop based on threshold (preview)
    cols_above = (miss_pct[miss_pct >= drop_cols_thresh].index.tolist()) if drop_cols else []
    preview_df = raw_df.copy()
    preview_audit = {"initial_rows": len(raw_df), "initial_cols": len(raw_df.columns)}

    # preview drop columns
    if cols_above:
        preview_df = preview_df.drop(columns=cols_above)
        preview_audit["cols_dropped_by_threshold"] = cols_above

    # preview row drops
    if row_drop_mode == "Drop rows with any missing value":
        rows_after = len(preview_df.dropna())
        preview_audit["rows_dropped_by_any_missing"] = int(len(preview_df) - rows_after)
        preview_df_preview = preview_df.dropna()
    elif row_drop_mode == "Drop rows with missing in selected columns" and cols_to_check:
        before = len(preview_df)
        preview_df_preview = preview_df.dropna(subset=cols_to_check)
        preview_audit["rows_dropped_by_selected_cols"] = int(before - len(preview_df_preview))
    else:
        preview_df_preview = preview_df

    st.markdown("**Preview shape after applying chosen options**")
    st.write(f"Rows: {preview_df_preview.shape[0]}  —  Columns: {preview_df_preview.shape[1]}")
    st.markdown("**What would be dropped (preview)**")
    if cols_above:
        st.write(f"- Columns to be dropped (missing ≥ {drop_cols_thresh:.2%}):")
        st.write(cols_above)
    else:
        st.write("- No columns will be dropped by threshold.")

    if row_drop_mode == "Don't drop rows":
        st.write("- Rows will not be dropped.")
    elif row_drop_mode == "Drop rows with any missing value":
        st.write(f"- Rows with any missing value will be dropped. Count preview: {preview_audit.get('rows_dropped_by_any_missing',0)}")
    else:
        st.write(f"- Rows with missing in selected columns {cols_to_check} will be dropped. Count preview: {preview_audit.get('rows_dropped_by_selected_cols',0)}")

    st.markdown("**Preview sample**")
    st.dataframe(preview_df_preview.head(50))

    st.markdown("---")

    # --- Actions: Apply, Save, Rollback
    st.subheader("Apply changes")
    apply_col, apply_notify = st.columns([1, 3])
    with apply_col:
        if st.button("Apply & Save cleaned dataset"):
            # perform actual operations and save cleaned df to session
            cleaned = raw_df.copy()

            # drop columns if enabled
            dropped_columns = []
            if drop_cols and cols_above:
                dropped_columns = cols_above
                cleaned = cleaned.drop(columns=dropped_columns)

            # drop rows according to mode
            dropped_rows_count = 0
            if row_drop_mode == "Drop rows with any missing value":
                before = len(cleaned)
                cleaned = cleaned.dropna()
                dropped_rows_count = before - len(cleaned)
            elif row_drop_mode == "Drop rows with missing in selected columns" and cols_to_check:
                before = len(cleaned)
                cleaned = cleaned.dropna(subset=cols_to_check)
                dropped_rows_count = before - len(cleaned)

            # save to session
            st.session_state[SESSION_CLEAN] = cleaned
            st.session_state[SESSION_CLEAN_META] = {
                "cols_dropped_by_threshold": dropped_columns,
                "rows_dropped_count": int(dropped_rows_count),
                "method": "drop",
                "drop_cols_threshold": float(drop_cols_thresh),
                "row_drop_mode": row_drop_mode,
                "row_drop_columns": cols_to_check
            }

            st.success(f"Applied cleaning. Cleaned dataset saved to session as '{SESSION_CLEAN}'.")
            st.info(f"Dropped {len(dropped_columns)} columns and {dropped_rows_count} rows.")

    with apply_notify:
        st.write("After applying you can preview the cleaned dataset and download it. Use rollback to remove the cleaned dataset from session if you need to retry.")

    st.markdown("---")

    # --- Show current cleaned dataset if present
    st.subheader("Current cleaned dataset (if any)")
    if SESSION_CLEAN in st.session_state:
        cd = st.session_state[SESSION_CLEAN]
        meta = st.session_state.get(SESSION_CLEAN_META, {})
        st.write(f"Cleaned shape: {cd.shape[0]} rows × {cd.shape[1]} columns")
        if meta:
            st.write("Audit:")
            st.json(meta)
        st.dataframe(cd.head(100))
        csv = cd.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned CSV", csv, file_name="cleaned_dataset.csv", mime="text/csv")

        if st.button("Rollback cleaned dataset (remove from session)"):
            del st.session_state[SESSION_CLEAN]
            if SESSION_CLEAN_META in st.session_state:
                del st.session_state[SESSION_CLEAN_META]
            st.success("Rolled back cleaned dataset from session.")
    else:
        st.info("No cleaned dataset in session. Use 'Apply & Save cleaned dataset' to create one.")

    st.markdown("---")
    st.caption("This cleaning page intentionally keeps behavior simple and reproducible: dropping missing values is deterministic and avoids complex imputation issues. If you want imputation later, we can add controlled options (median/group-median/knn/mice) with validation tools.")
