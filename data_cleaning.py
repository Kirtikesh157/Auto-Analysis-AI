# data_cleaning.py
import streamlit as st
import pandas as pd
import numpy as np
from copy import deepcopy

# Optional model-based imputers
try:
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

SESSION_RAW = "__uploaded_df__"
SESSION_CLEAN = "__cleaned_df__"
SESSION_CLEAN_META = "__cleaned_meta__"

def _missing_summary(df: pd.DataFrame):
    miss = df.isna().sum()
    pct = miss / len(df)
    summary = pd.DataFrame({
        "missing_count": miss,
        "missing_pct": pct
    }).sort_values("missing_pct", ascending=False)
    return summary

def _default_strategy_for_series(s: pd.Series):
    if pd.api.types.is_numeric_dtype(s):
        return "median"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "ffill"
    if pd.api.types.is_bool_dtype(s):
        return "mode"
    # object / categorical / text
    nunique = s.nunique(dropna=True)
    if nunique <= 20:
        return "mode"
    return "missing_token"

def _apply_strategy_to_series(s: pd.Series, strategy: str, constant_value=None, groupby=None, df=None):
    # returns new_series, info dict (what was done)
    info = {"strategy": strategy}
    orig_na = s.isna()
    new_s = s.copy()
    if strategy == "leave":
        return new_s, info
    if strategy == "drop":
        return None, info
    if strategy == "median":
        med = s.median()
        new_s = s.fillna(med)
        info["filled_with"] = med
    elif strategy == "mean":
        m = s.mean()
        new_s = s.fillna(m)
        info["filled_with"] = m
    elif strategy == "mode":
        try:
            mode = s.mode(dropna=True)
            modev = mode.iloc[0] if len(mode)>0 else ""
        except Exception:
            modev = ""
        new_s = s.fillna(modev)
        info["filled_with"] = modev
    elif strategy == "constant":
        new_s = s.fillna(constant_value)
        info["filled_with"] = constant_value
    elif strategy == "missing_token":
        new_s = s.fillna("__MISSING__")
        info["filled_with"] = "__MISSING__"
    elif strategy == "ffill":
        new_s = s.fillna(method="ffill")
        info["filled_with"] = "ffill"
    elif strategy == "bfill":
        new_s = s.fillna(method="bfill")
        info["filled_with"] = "bfill"
    elif strategy == "interpolate":
        try:
            new_s = s.interpolate(method="linear")
            info["filled_with"] = "interpolate"
        except Exception:
            new_s = s
            info["error"] = "interpolate_failed"
    elif strategy == "knn" and SKLEARN_AVAILABLE:
        # KNN imputer needs numeric matrix: user should pick numeric-only columns in UI
        raise NotImplementedError("knn should be applied at dataframe-level using sklearn's KNNImputer")
    elif strategy == "mice" and SKLEARN_AVAILABLE:
        raise NotImplementedError("mice requires iterative imputer at dataframe-level")
    else:
        # fallback
        new_s = s
        info["error"] = f"strategy_{strategy}_not_implemented"
    info["n_filled"] = int(orig_na.sum() - new_s.isna().sum())
    return new_s, info

def render_data_cleaning():
    st.header("Data Cleaning")
    if SESSION_RAW not in st.session_state:
        st.warning("No dataset loaded. Go to Data Upload and load a dataset first.")
        return

    raw_df = st.session_state[SESSION_RAW]
    # ensure we work on a copy
    if SESSION_CLEAN in st.session_state:
        clean_df = st.session_state[SESSION_CLEAN].copy()
    else:
        clean_df = raw_df.copy()

    st.markdown("### Missing values overview")
    summary = _missing_summary(raw_df)
    st.dataframe(summary)

    # give quick stats
    total_missing = int(summary["missing_count"].sum())
    st.markdown(f"**Total missing cells:** {total_missing}")

    # threshold to drop columns with too many missing values
    drop_thresh = st.slider("Drop columns with missing % >= ", 0.0, 1.0, 0.5, step=0.05, format="%.2f")
    cols_high_missing = summary[summary["missing_pct"] >= drop_thresh].index.tolist()
    if cols_high_missing:
        st.warning(f"Columns above threshold ({drop_thresh*100:.0f}%): {', '.join(cols_high_missing[:6])}" + (", ..." if len(cols_high_missing)>6 else ""))

    st.markdown("---")
    st.markdown("### Per-column strategies")
    # Prepare a stateful dict to store chosen strategies
    if "cleaning_plans" not in st.session_state:
        st.session_state.cleaning_plans = {}

    cols = raw_df.columns.tolist()
    for col in cols:
        with st.expander(f"{col} — {str(raw_df[col].dtype)} — missing {int(summary.loc[col,'missing_count'])} ({summary.loc[col,'missing_pct']*100:.1f}%)", expanded=False):
            st.write("Preview:")
            st.dataframe(raw_df[[col]].head(5))
            default = _default_strategy_for_series(raw_df[col])
            # read existing choice
            prev = st.session_state.cleaning_plans.get(col, {}).get("strategy", default)
            strategy = st.selectbox(f"Strategy for `{col}`", options=["leave","drop","median","mean","mode","constant","missing_token","ffill","bfill","interpolate","knn","mice"], index=["leave","drop","median","mean","mode","constant","missing_token","ffill","bfill","interpolate","knn","mice"].index(prev) if prev in ["leave","drop","median","mean","mode","constant","missing_token","ffill","bfill","interpolate","knn","mice"] else 0, key=f"strat_{col}")
            const_val = None
            if strategy == "constant":
                const_val = st.text_input(f"Constant value for `{col}`", key=f"const_{col}")
            create_indicator = st.checkbox("Create missing indicator column", value=st.session_state.cleaning_plans.get(col, {}).get("indicator", False), key=f"ind_{col}")
            # group-by option
            groupby_opt = None
            st.session_state.cleaning_plans[col] = {"strategy": strategy, "constant": const_val, "indicator": create_indicator, "groupby": groupby_opt}

    st.markdown("---")
    st.markdown("### Actions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Preview changes"):
            # build a preview dataframe applying strategies to copies (no session write)
            preview_df = clean_df.copy()
            change_log = {}
            for col, plan in st.session_state.cleaning_plans.items():
                strat = plan["strategy"]
                const = plan.get("constant")
                if strat in ("knn","mice"):
                    # skip dataframe-level methods in preview (show notice)
                    change_log[col] = {"note":"requires dataframe-level imputer (skipped in preview)"}
                    continue
                new_col, info = _apply_strategy_to_series(preview_df[col], strat, constant_value=const, df=preview_df)
                if new_col is None:
                    preview_df.drop(columns=[col], inplace=True)
                    change_log[col] = {"action":"drop"}
                else:
                    preview_df[col] = new_col
                    change_log[col] = info
                if plan.get("indicator"):
                    preview_df[f"{col}__missing"] = preview_df[col].isna().astype(int)
            st.write("Preview head (first 10 rows):")
            st.dataframe(preview_df.head(10))
            st.write("Preview change log (per column):")
            st.json(change_log)

    with c2:
        if st.button("Apply selected strategies"):
            # Apply column-level strategies
            new_df = clean_df.copy()
            applied_log = {}
            # optionally drop columns above threshold first
            for drop_col in cols_high_missing:
                if drop_col in new_df.columns:
                    new_df.drop(columns=[drop_col], inplace=True)
                    applied_log[drop_col] = {"action":"dropped_due_missing_pct"}
            # apply per-column
            for col, plan in st.session_state.cleaning_plans.items():
                if col not in new_df.columns:
                    continue
                strat = plan["strategy"]
                const = plan.get("constant")
                if strat == "knn" or strat == "mice":
                    # skip here; handle after loop at dataframe-level
                    continue
                new_col, info = _apply_strategy_to_series(new_df[col], strat, constant_value=const, df=new_df)
                if new_col is None:
                    if col in new_df.columns:
                        new_df.drop(columns=[col], inplace=True)
                        applied_log[col] = {"action":"dropped_by_plan"}
                else:
                    new_df[col] = new_col
                    applied_log[col] = info
                if plan.get("indicator"):
                    new_df[f"{col}__missing"] = raw_df[col].isna().astype(int)  # indicator based on original raw
            # handle knn/mice if requested:
            # KNN/Iterative imputers usually need numeric columns; find columns that requested knn/mice
            knn_cols = [col for col, p in st.session_state.cleaning_plans.items() if p["strategy"]=="knn" and col in new_df.columns]
            mice_cols = [col for col, p in st.session_state.cleaning_plans.items() if p["strategy"]=="mice" and col in new_df.columns]
            if SKLEARN_AVAILABLE and (knn_cols or mice_cols):
                # Select numeric columns for imputation; for simplicity restrict to numeric subset
                num_cols = new_df.select_dtypes(include=["number"]).columns.tolist()
                impute_cols = [c for c in num_cols if c in (knn_cols + mice_cols)]
                if impute_cols:
                    # Prepare imputer for knn (if any)
                    if knn_cols:
                        try:
                            knn_imp = KNNImputer(n_neighbors=5)
                            new_df[impute_cols] = knn_imp.fit_transform(new_df[impute_cols])
                            for c in impute_cols:
                                applied_log[c] = {"strategy":"knn_imputed"}
                        except Exception as e:
                            st.error(f"KNN imputation failed: {e}")
                    # MICE / Iterative
                    if mice_cols:
                        try:
                            mice_imp = IterativeImputer(max_iter=10, random_state=0)
                            new_df[impute_cols] = mice_imp.fit_transform(new_df[impute_cols])
                            for c in impute_cols:
                                applied_log[c] = {"strategy":"mice_imputed"}
                        except Exception as e:
                            st.error(f"Iterative imputation failed: {e}")
                else:
                    st.warning("No numeric columns available for KNN/MICE imputation; skipping those strategies.")
            elif (knn_cols or mice_cols) and not SKLEARN_AVAILABLE:
                st.warning("scikit-learn not available in environment — KNN/MICE skipped. Install scikit-learn to enable.")

            # Save cleaned df to session state and record meta
            st.session_state[SESSION_CLEAN] = new_df
            st.session_state[SESSION_CLEAN_META] = {"applied_log": applied_log, "n_rows": new_df.shape[0], "n_cols": new_df.shape[1]}
            st.success("Cleaning applied and saved to session as '__cleaned_df__'.")

    with c3:
        if st.button("Quick Clean (safe defaults)"):
            # safe defaults: numeric->median, categorical->mode, create indicators
            qc_df = raw_df.copy()
            qc_log = {}
            for col in qc_df.columns:
                s = qc_df[col]
                if s.isna().sum() == 0:
                    continue
                if pd.api.types.is_numeric_dtype(s):
                    med = s.median()
                    qc_df[col] = s.fillna(med)
                    qc_log[col] = {"strategy":"median","filled_with":med}
                elif pd.api.types.is_datetime64_any_dtype(s):
                    qc_df[col] = s.fillna(method="ffill")
                    qc_log[col] = {"strategy":"ffill"}
                elif pd.api.types.is_bool_dtype(s):
                    # fill with mode
                    try:
                        m = s.mode().iloc[0]
                    except Exception:
                        m = False
                    qc_df[col] = s.fillna(m)
                    qc_log[col] = {"strategy":"mode","filled_with":m}
                else:
                    # categorical/text
                    try:
                        modev = s.mode().iloc[0]
                    except Exception:
                        modev = "__MISSING__"
                    qc_df[col] = s.fillna(modev)
                    qc_log[col] = {"strategy":"mode_or_missing","filled_with":modev}
                # indicator
                qc_df[f"{col}__missing"] = s.isna().astype(int)
            st.session_state[SESSION_CLEAN] = qc_df
            st.session_state[SESSION_CLEAN_META] = {"applied_log": qc_log, "method":"quick_clean", "n_rows": qc_df.shape[0], "n_cols": qc_df.shape[1]}
            st.success("Quick-clean applied and saved to session as '__cleaned_df__'.")

    with c4:
        if st.button("Rollback to original raw"):
            if SESSION_CLEAN in st.session_state:
                del st.session_state[SESSION_CLEAN]
            if SESSION_CLEAN_META in st.session_state:
                del st.session_state[SESSION_CLEAN_META]
            st.success("Rolled back cleaned dataset from session. Raw dataset remains in '__uploaded_df__'.")

    st.markdown("---")
    st.markdown("### Current cleaned dataset preview (if any)")
    if SESSION_CLEAN in st.session_state:
        st.dataframe(st.session_state[SESSION_CLEAN].head(50))
        # download
        csv = st.session_state[SESSION_CLEAN].to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned CSV", csv, file_name="cleaned_dataset.csv", mime="text/csv")
    else:
        st.info("No cleaned dataset in session. Apply cleaning or use Quick Clean.")

    # End of render
