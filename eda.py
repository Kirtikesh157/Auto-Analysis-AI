# eda.py
"""
Advanced interactive EDA (Mode B) for AutoAnalyst AI — full, ID-safe version.

Expose: render_advanced_eda()

Reads from:
  - st.session_state["__cleaned_df__"] (preferred)
  - fallback: st.session_state["__uploaded_df__"]

Sections (expandable):
  1) Overview & Missingness
  2) Visual EDA (distributions & categories)
  3) Statistical tests (t-test, ANOVA, Chi-square)
  4) Correlations & VIF
  5) Segment discovery (KMeans)
  6) Outliers & anomalies
  7) Business heuristics (Pareto, categorical imbalance) — excludes ID-like columns
  8) Narrative summary (one paragraph)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# optional libraries (import safely)
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# session keys used by your app
SESSION_RAW = "__uploaded_df__"
SESSION_CLEAN = "__cleaned_df__"

# ---------------- Helper functions ----------------
def _get_df():
    """Return (df, used_clean_flag)"""
    if SESSION_CLEAN in st.session_state:
        return st.session_state[SESSION_CLEAN].copy(), True
    if SESSION_RAW in st.session_state:
        return st.session_state[SESSION_RAW].copy(), False
    return None, False

def _numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def _cat_cols(df):
    return df.select_dtypes(include=["object","category","bool"]).columns.tolist()

def _missing_summary(df):
    miss = df.isna().sum()
    pct = (miss / len(df)).round(4)
    return pd.DataFrame({"missing_count": miss, "missing_pct": pct}).sort_values("missing_pct", ascending=False)

def _is_id_like(series, n_rows, uniq_ratio_threshold=0.95):
    """
    Heuristic to detect ID-like (primary key or near-unique) columns.
    - perfect uniqueness OR high uniqueness ratio (>= uniq_ratio_threshold for reasonably sized datasets)
    - string heuristics: long average token length or uuid-like pattern
    """
    try:
        n = n_rows
        non_null = series.dropna()
        if n <= 1:
            return False
        nunique = non_null.nunique(dropna=True)
        # perfect uniqueness
        if nunique == n:
            return True
        # high uniqueness ratio (only for datasets of reasonable size)
        if n > 50 and (nunique / float(n) >= uniq_ratio_threshold):
            return True
        # string heuristics for e.g., UUIDs/hashes
        if series.dtype == 'object' and non_null.size > 0:
            sample = non_null.astype(str).sample(min(50, len(non_null)), random_state=0)
            avg_len = sample.map(len).mean()
            if avg_len > 20:
                return True
            if sample.str.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-').any():
                return True
    except Exception:
        return False
    return False

def _pareto_safe(df, col, top_k=10):
    """
    Safer Pareto check:
    - skip missing/empty
    - skip id-like columns
    - require numeric dtype, positive total, and nonzero variance
    Returns (vc_series, top_share, skip_msg_or_None)
    """
    if col not in df.columns:
        return None, None, f"column '{col}' not present"
    s = df[col].dropna()
    if s.empty:
        return None, None, "no non-null values"
    if _is_id_like(df[col], len(df)):
        return None, None, "skipped: id-like or near-unique column"
    if not pd.api.types.is_numeric_dtype(s):
        return None, None, "skipped: not numeric"
    total = s.sum()
    if total <= 0:
        return None, None, "skipped: non-positive total"
    if s.std() == 0:
        return None, None, "skipped: no variance"
    top = s.sort_values(ascending=False).head(top_k)
    top_share = float(top.sum() / total)
    return top, top_share, None

def _safe_corr(df):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    return num.corr()

def _compute_vif(df):
    if not STATSMODELS_AVAILABLE:
        return None, "statsmodels not installed"
    num = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num.shape[1] < 2:
        return None, "not enough numeric cols for VIF"
    # drop zero-variance columns
    num = num.loc[:, num.std() > 0]
    if num.shape[1] < 2:
        return None, "not enough variable numeric columns for VIF after dropping zero-variance"
    try:
        X = sm.add_constant(num)
        vif_list = []
        for i, col in enumerate(num.columns):
            try:
                vif = variance_inflation_factor(X.values, i+1)  # +1 because const at 0
            except Exception:
                vif = np.nan
            vif_list.append({"feature": col, "vif": float(vif) if not pd.isna(vif) else None})
        return pd.DataFrame(vif_list).sort_values("vif", ascending=False), None
    except Exception as e:
        return None, str(e)

def _t_test(df, numeric_col, by_col):
    if not SCIPY_AVAILABLE:
        return None, "scipy not installed"
    tmp = df.dropna(subset=[numeric_col, by_col])
    groups = tmp.groupby(by_col)[numeric_col]
    keys = list(groups.groups.keys())
    if len(keys) != 2:
        return None, "requires exactly 2 groups for t-test"
    a = groups.get_group(keys[0])
    b = groups.get_group(keys[1])
    try:
        stat, p = stats.ttest_ind(a, b, nan_policy='omit', equal_var=False)
        return {"group1": keys[0], "group2": keys[1], "stat": float(stat), "pvalue": float(p)}, None
    except Exception as e:
        return None, str(e)

def _anova_test(df, numeric_col, cat_col):
    if not SCIPY_AVAILABLE:
        return None, "scipy not installed"
    try:
        groups = [g.dropna() for _, g in df.groupby(cat_col)[numeric_col]]
        if len(groups) < 2:
            return None, "not enough groups for ANOVA"
        stat, p = stats.f_oneway(*groups)
        return {"f_stat": float(stat), "pvalue": float(p)}, None
    except Exception as e:
        return None, str(e)

def _chi2_test(df, col1, col2):
    if not SCIPY_AVAILABLE:
        return None, "scipy not installed"
    try:
        table = pd.crosstab(df[col1], df[col2])
        if table.size == 0:
            return None, "insufficient data for chi-square"
        chi2, p, dof, ex = stats.chi2_contingency(table)
        return {"chi2": float(chi2), "pvalue": float(p), "dof": int(dof)}, None
    except Exception as e:
        return None, str(e)

def _cluster_preview(df, n_clusters=3, sample_limit=5000):
    if not SKLEARN_AVAILABLE:
        return None, "sklearn not installed"
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[0] == 0 or num.shape[1] == 0:
        return None, "no numeric data to cluster"
    if num.shape[0] > sample_limit:
        num = num.sample(sample_limit, random_state=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(num)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    try:
        pca = PCA(n_components=2, random_state=0)
        proj = pca.fit_transform(X)
        proj_df = pd.DataFrame({"pc1": proj[:,0], "pc2": proj[:,1], "cluster": labels}, index=num.index)
        return proj_df, None
    except Exception:
        # fallback: return label counts only
        return pd.Series(labels).value_counts().to_frame("count"), None

def _iqr_outliers(series):
    s = series.dropna()
    if s.empty:
        return pd.Series([], dtype=s.dtype)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series([], dtype=s.dtype)
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    return s[(s>upper) | (s<lower)]

def _build_narrative(findings):
    """
    Compose a one-paragraph narrative from prioritized findings.
    Rules:
      - Only include correlations with |r| >= 0.5
      - Only include categorical imbalance if top category >= 40%
      - Only include Pareto if top_k share >= 30%
      - Only include missingness mention if >= 1%
      - Only include outlier statement when there are >0 IQR outliers
      - ID-like columns are filtered upstream
    """
    parts = []

    # dataset shape
    if "shape" in findings:
        parts.append(f"The dataset contains {findings['shape'][0]:,} rows and {findings['shape'][1]:,} columns.")

    # missingness (top)
    if "missing" in findings and findings["missing"]:
        top_missing = findings["missing"][0]
        if top_missing.get("pct", 0) >= 0.01:  # mention only when >=1%
            parts.append(f"Highest missingness is in `{top_missing['column']}` ({top_missing['pct']:.1%} missing).")

    # strong correlations
    if "correlation" in findings and findings["correlation"]:
        corr = findings["correlation"][0]
        parts.append(f"Notably, `{corr['a']}` and `{corr['b']}` show a strong correlation ({corr['val']:+.2f}).")

    # categorical imbalance
    if "cat_imbalance" in findings and findings["cat_imbalance"]:
        cat = findings["cat_imbalance"][0]
        parts.append(f"Category `{cat['column']}` is dominated by `{cat['top']}` ({cat['pct']:.1%}), which may affect modeling fairness.")

    # outlier mention
    if "outlier" in findings and findings["outlier"]:
        out = findings["outlier"][0]
        if "max" in out:
            parts.append(f"`{out['column']}` contains extreme values (e.g., max ≈ {out['max']:,}). Consider inspecting these cases.")

    # Pareto
    if "pareto" in findings and findings["pareto"]:
        p = findings["pareto"][0]
        parts.append(f"Top {p['top_k']} entries of `{p['column']}` contribute {p['share']:.1%} of its total — a notable Pareto effect.")

    if not parts:
        return "No major issues detected: the dataset appears well-formed for exploratory analysis."

    paragraph = " ".join(parts)
    return paragraph

# ---------------- Render function ----------------
def render_eda():
    
    st.header("Advanced EDA — Interactive (Mode B)")

    df, used_clean = _get_df()
    if df is None:
        st.error("No dataset found. Upload and/or clean data first.")
        return

    st.write(f"Using {'cleaned' if used_clean else 'raw'} dataset — {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    findings = {}
    findings["shape"] = df.shape

    # 1) Overview & Missingness
    with st.expander("1) Overview & Missingness", expanded=True):
        st.subheader("Top rows")
        st.dataframe(df.head(10))
        st.subheader("Missingness summary")
        miss_df = _missing_summary(df)
        st.dataframe(miss_df.style.format({"missing_pct":"{:.2%}"}))
        top_missing = miss_df[miss_df["missing_pct"]>0].reset_index().to_dict("records")
        findings["missing"] = [{"column": r["index"], "pct": float(r["missing_pct"])} for r in top_missing[:3]]

    # 2) Visual EDA
    with st.expander("2) Visual EDA (distributions & categories)", expanded=False):
        nums = _numeric_cols(df)
        cats = _cat_cols(df)
        st.write("Numeric columns:", nums if nums else "None")
        st.write("Categorical columns:", cats if cats else "None")

        sel_col = st.selectbox("Select a column to visualize", options=df.columns.tolist())
        if sel_col in nums:
            s = df[sel_col].dropna()
            if s.empty:
                st.info("Column has no non-null values.")
            else:
                fig, axes = plt.subplots(1,2, figsize=(12,4))
                sns.histplot(s, kde=True, ax=axes[0])
                axes[0].set_title(f"Histogram: {sel_col}")
                sns.boxplot(x=s, ax=axes[1])
                axes[1].set_title("Boxplot")
                st.pyplot(fig)
                skew = float(s.skew())
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                upper = q3 + 1.5*iqr if iqr>0 else s.max()
                out_cnt = int((s>upper).sum()) if iqr>0 else 0
                st.write(f"Skewness: {skew:.2f}. Upper outliers (IQR rule): {out_cnt}.")
                findings.setdefault("outlier", []).append({"column": sel_col, "max": float(s.max())})
        else:
            vc = df[sel_col].value_counts().iloc[:40]
            fig, ax = plt.subplots(figsize=(8,4))
            vc.plot(kind="bar", ax=ax)
            ax.set_title(f"Value counts: {sel_col}")
            st.pyplot(fig)
            if not vc.empty:
                top = vc.index[0]
                pct = float(vc.iloc[0]/vc.sum())
                st.write(f"Top category: `{top}` ({pct:.1%}).")
                # ONLY record categorical imbalance candidate if NOT ID-like AND pct >= 40%
                if (not _is_id_like(df[sel_col], len(df))) and (pct >= 0.40):
                    findings.setdefault("cat_imbalance", []).append({"column": sel_col, "top": str(top), "pct": pct})

    # 3) Statistical tests
    with st.expander("3) Statistical tests (t-test, ANOVA, Chi-square)", expanded=False):
        if not SCIPY_AVAILABLE:
            st.warning("scipy not installed — statistical tests disabled.")
        else:
            st.markdown("t-test (requires binary categorical column)")
            num_opts = _numeric_cols(df)
            cat_binary = [c for c in _cat_cols(df) if df[c].nunique(dropna=True) == 2]
            if num_opts and cat_binary:
                t_num = st.selectbox("Numeric for t-test", options=num_opts, index=0, key="t_num")
                t_cat = st.selectbox("Binary categorical for t-test", options=cat_binary, index=0, key="t_cat")
                if st.button("Run t-test"):
                    res, err = _t_test(df, t_num, t_cat)
                    if err:
                        st.info(err)
                    else:
                        st.write(f"t-stat = {res['stat']:.3f}, p-value = {res['pvalue']:.3g} (groups: {res['group1']} vs {res['group2']})")
            else:
                st.info("Not enough suitable columns for t-test (need numeric + binary categorical).")

            st.markdown("ANOVA (numeric across multiple categories)")
            a_nums = _numeric_cols(df)
            a_cats = [c for c in _cat_cols(df) if df[c].nunique(dropna=True)>1]
            if a_nums and a_cats:
                an_num = st.selectbox("Numeric (ANOVA)", options=a_nums, index=0, key="an_num")
                an_cat = st.selectbox("Categorical (ANOVA)", options=a_cats, index=0, key="an_cat")
                if st.button("Run ANOVA"):
                    an_res, an_err = _anova_test(df, an_num, an_cat)
                    if an_err:
                        st.info(an_err)
                    else:
                        st.write(f"ANOVA F-stat = {an_res['f_stat']:.3f}, p-value = {an_res['pvalue']:.3g}")

            st.markdown("Chi-square (two categorical cols)")
            catlist = _cat_cols(df)
            if len(catlist) >= 2:
                ch_a = st.selectbox("Categorical A", options=catlist, index=0, key="chi_a")
                ch_b = st.selectbox("Categorical B", options=[c for c in catlist if c!=ch_a], index=0 if len(catlist)>1 else 0, key="chi_b")
                if st.button("Run Chi-square"):
                    ch_res, ch_err = _chi2_test(df, ch_a, ch_b)
                    if ch_err:
                        st.info(ch_err)
                    else:
                        st.write(f"Chi2 = {ch_res['chi2']:.3f}, p-value = {ch_res['pvalue']:.3g}, dof = {ch_res['dof']}")
            else:
                st.info("Need at least two categorical columns for Chi-square.")

    # 4) Correlation & VIF
    with st.expander("4) Correlations & VIF", expanded=False):
        corr = _safe_corr(df)
        if corr is None:
            st.info("Not enough numeric columns for correlation analysis.")
        else:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="vlag", center=0, ax=ax)
            st.pyplot(fig)
            # record strongest pair only if meaningfully strong (|r| >= 0.5)
            masked = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            if masked.stack().empty:
                st.info("No pairwise correlations found.")
            else:
                toppair = masked.stack().idxmax()
                a, b = toppair
                val = corr.loc[a, b]
                if abs(val) >= 0.5:
                    findings.setdefault("correlation", []).append({"a": a, "b": b, "val": float(val)})
                    st.write(f"Strongest correlation: **{a}** vs **{b}** => {val:.2f}")
                else:
                    st.info(f"No strong correlations detected (strong threshold |r| >= 0.50). Top observed correlation: {a} vs {b} => {val:.2f}")

            # VIF
            vif_df, vif_err = _compute_vif(df)
            if vif_err:
                st.info(vif_err)
            else:
                st.subheader("VIF (multicollinearity)")
                st.dataframe(vif_df)
                high = vif_df[vif_df["vif"]>10]
                if not high.empty:
                    st.warning("High multicollinearity detected (VIF > 10).")
                    st.dataframe(high)

    # 5) Clustering / segments
    with st.expander("5) Segment discovery (KMeans, optional)", expanded=False):
        if not SKLEARN_AVAILABLE:
            st.info("sklearn not installed — clustering disabled.")
        else:
            k = st.slider("Number of clusters (KMeans)", 2, 8, 3)
            sample_limit = st.number_input("Sample limit for clustering", min_value=200, max_value=20000, value=2000, step=100)
            if st.button("Run clustering"):
                res, err = _cluster_preview(df, n_clusters=k, sample_limit=sample_limit)
                if err:
                    st.info(err)
                else:
                    if isinstance(res, pd.DataFrame) and "pc1" in res.columns:
                        fig, ax = plt.subplots(figsize=(8,6))
                        sns.scatterplot(data=res, x="pc1", y="pc2", hue="cluster", palette="tab10", ax=ax)
                        st.pyplot(fig)
                        st.write(res["cluster"].value_counts().sort_index())
                    else:
                        st.write(res)

    # 6) Outliers & anomalies
    with st.expander("6) Outliers & anomalies", expanded=False):
        st.markdown("Upper outliers by numeric column (IQR rule)")
        numeric = _numeric_cols(df)
        out_counts = {}
        for c in numeric:
            s = df[c].dropna()
            if s.empty: continue
            out = _iqr_outliers(s)
            out_counts[c] = int(out.shape[0]) if not out.empty else 0
        if out_counts:
            out_df = pd.DataFrame.from_dict(out_counts, orient="index", columns=["upper_outliers"]).sort_values("upper_outliers", ascending=False)
            st.dataframe(out_df.head(20))
            topcol = out_df["upper_outliers"].idxmax()
            if out_df.loc[topcol, "upper_outliers"] > 0:
                findings.setdefault("outlier", []).append({"column": topcol, "max": float(df[topcol].max())})
        else:
            st.info("No numeric outliers detected (or insufficient data).")

    # 7) Business heuristics (Pareto & categorical imbalance) — safe
    with st.expander("7) Business heuristics (Pareto, top contributors)", expanded=False):
        st.markdown("Pareto (top contributors) — ID-like columns are excluded automatically")
        numeric_cols = _numeric_cols(df)
        meaningful = [c for c in numeric_cols if not _is_id_like(df[c], len(df))]
        if not meaningful:
            st.info("No meaningful numeric columns available for Pareto checks (either ID-like or no variance).")
        else:
            pc = st.selectbox("Numeric column for Pareto check", options=meaningful, index=0)
            if pc:
                vc, share, msg = _pareto_safe(df, pc, top_k=10)
                PARETO_THRESHOLD = 0.30
                if msg:
                    st.info(msg)
                else:
                    if share >= PARETO_THRESHOLD:
                        st.write(f"Top 10 entries contribute {share:.1%} of `{pc}` total — meaningful Pareto effect.")
                        findings.setdefault("pareto", []).append({"column": pc, "top_k": 10, "share": share})
                    else:
                        st.info(f"Top 10 entries contribute {share:.1%} of `{pc}` total (below Pareto threshold {PARETO_THRESHOLD:.0%}).")

        st.markdown("Categorical imbalance (ID-like columns excluded)")
        cat_candidates = [c for c in _cat_cols(df) if not _is_id_like(df[c], len(df))]
        if not cat_candidates:
            st.info("No categorical columns (excluding ID-like) for imbalance checks.")
        else:
            IMBALANCE_THRESHOLD = 0.40
            for c in cat_candidates[:20]:
                vc = df[c].value_counts(normalize=True)
                if vc.empty: continue
                top_pct = float(vc.iloc[0])
                if top_pct >= IMBALANCE_THRESHOLD:
                    st.warning(f"`{c}` dominated by `{vc.index[0]}` ({top_pct:.1%}) — strong imbalance.")
                    findings.setdefault("cat_imbalance", []).append({"column": c, "top": str(vc.index[0]), "pct": top_pct})

    # 8) Narrative summary
    with st.expander("8) Narrative summary (one-paragraph)", expanded=True):
        paragraph = _build_narrative(findings)
        st.subheader("Key narrative")
        st.write(paragraph)
        st.markdown("**Copyable summary**")
        st.code(paragraph)
        if st.button("Export findings as JSON"):
            try:
                import json
                payload = {"findings": findings, "n_rows": df.shape[0], "n_cols": df.shape[1]}
                st.download_button("Download findings JSON", data=json.dumps(payload, default=str, indent=2).encode("utf-8"), file_name="eda_findings.json", mime="application/json")
            except Exception as e:
                st.error(f"Failed to export JSON: {e}")

    st.success("Interactive advanced EDA complete.")
