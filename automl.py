# automl.py
"""
Simplified AutoML page for AutoAnalyst AI.

Design goals:
- Simple, fast 3-step flow: select target -> pick model -> train (one-click)
- Sensible defaults for speedy runs
- Advanced options collapsed in an expander
- Optional Predict and SHAP features (toggled)
- Robust across sklearn versions (OneHotEncoder compatibility; manual RMSE)
- Saves trained pipeline to st.session_state["__trained_model__"]
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_absolute_error, r2_score

# models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

# optional xgboost/shap availability (soft)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# session keys
SESSION_RAW = "__uploaded_df__"
SESSION_CLEAN = "__cleaned_df__"
SESSION_MODEL = "__trained_model__"

# ------------------ Helpers ------------------
def _get_df():
    if SESSION_CLEAN in st.session_state:
        return st.session_state[SESSION_CLEAN], True
    if SESSION_RAW in st.session_state:
        return st.session_state[SESSION_RAW], False
    return None, False

def _onehot_encoder_compat():
    from sklearn.preprocessing import OneHotEncoder as _OHE
    kwargs = {"handle_unknown": "ignore"}
    try:
        _ = _OHE(**{"sparse_output": False})
        kwargs["sparse_output"] = False
    except TypeError:
        try:
            _ = _OHE(**{"sparse": False})
            kwargs["sparse"] = False
        except Exception:
            pass
    except Exception:
        try:
            _ = _OHE(**{"sparse": False})
            kwargs["sparse"] = False
        except Exception:
            pass
    return _OHE, kwargs

def _build_preprocessor(df, features):
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    OHE, ohe_kwargs = _onehot_encoder_compat()
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")), ("ohe", OHE(**ohe_kwargs))])

    preprocessor = ColumnTransformer([("num", numeric_transformer, num_cols), ("cat", categorical_transformer, cat_cols)], remainder="drop")
    return preprocessor, num_cols, cat_cols

def _detect_task(df, target):
    ser = df[target].dropna()
    if pd.api.types.is_numeric_dtype(ser):
        return "classification" if ser.nunique() <= 20 else "regression"
    return "classification"

def _regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    try:
        mae = float(mean_absolute_error(y_true, y_pred))
    except Exception:
        mae = float(np.mean(np.abs(y_true - y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}

def _classification_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    out = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                from sklearn.metrics import roc_auc_score
                out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:,1] if y_prob.ndim>1 else y_prob))
        except Exception:
            pass
    return out

# ------------------ UI ------------------
def render_automl():
    st.set_page_config(page_title="AutoML (simple)", layout="wide")
    st.title("AutoML — Quick & Simple")

    df, used_clean = _get_df()
    if df is None:
        st.warning("No dataset found. Upload or clean data first on the Upload/Clean page.")
        return
    st.write(f"Using {'cleaned' if used_clean else 'uploaded'} dataset — {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    # Step 1: choose target
    st.subheader("1) Choose target (what to predict)")
    target = st.selectbox("Target column", options=df.columns.tolist())
    if target is None:
        return
    st.caption("Tip: numeric target with many uniques → regression; else classification.")
    detected = _detect_task(df, target)
    task = st.radio("Detected task (override if needed)", options=["classification", "regression"], index=0 if detected=="classification" else 1)
    st.write(f"Using task: **{task}** (detected: {detected})")

    # Step 2: features
    st.subheader("2) Features")
    all_features = [c for c in df.columns if c != target]
    use_all = st.checkbox("Use all other columns as features (recommended)", value=True)
    if not use_all:
        features = st.multiselect("Select features", options=all_features, default=all_features[:min(6,len(all_features))])
    else:
        features = all_features
    if not features:
        st.error("No features selected.")
        return

    # Step 3: quick model selection
    st.subheader("3) Pick a quick model")
    # sensible default suggestions
    if task == "classification":
        model_options = ["Logistic Regression (fast)", "Random Forest (balanced)"]
        if XGBOOST_AVAILABLE:
            model_options.append("XGBoost (if installed)")
    else:
        model_options = ["Linear Regression (fast)", "Random Forest Regressor (balanced)"]
        if XGBOOST_AVAILABLE:
            model_options.append("XGBoost (if installed)")

    choice = st.selectbox("Model", options=model_options, index=0)
    st.write("Keep defaults for quick results. Expand Advanced to tune hyperparameters or enable CV/SHAP.")

    # Advanced options collapsed
    with st.expander("Advanced options (optional)"):
        test_size = st.slider("Test fraction", 0.05, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
        do_cv = st.checkbox("Run 3-fold CV (slower)", value=False)
        cv_folds = 3 if do_cv else 1
        enable_shap = st.checkbox("Enable SHAP explanations after training (may be slow)", value=False)
        shap_bg = st.slider("SHAP background rows", 50, 500, 100, step=50) if enable_shap else 0
    # if advanced not used, defaults:
    if "test_size" not in locals():
        test_size = 0.2
    if "random_state" not in locals():
        random_state = 42
    if "do_cv" not in locals():
        do_cv = False
        cv_folds = 1
    if "enable_shap" not in locals():
        enable_shap = False

    # Train button
    if st.button("Train (quick)"):
        # simple preparation
        working = df.dropna(subset=[target]).copy()
        X = working[features]
        y = working[target]

        preprocessor, num_cols, cat_cols = _build_preprocessor(working, features)

        # pick model
        model = None
        try:
            if "Logistic" in choice:
                model = LogisticRegression(max_iter=200)
            elif "Linear Regression" in choice:
                model = LinearRegression()
            elif "Random Forest" in choice:
                if task == "classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            elif "XGBoost" in choice and XGBOOST_AVAILABLE:
                model = xgb.XGBClassifier() if task=="classification" else xgb.XGBRegressor()
            else:
                # fallback
                model = RandomForestClassifier(n_estimators=100) if task=="classification" else RandomForestRegressor(n_estimators=100)
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
            return

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # split
        stratify_arg = y if (task=="classification" and y.nunique()>1) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=stratify_arg)
        except Exception as e:
            st.error(f"Train/test split failed: {e}")
            return

        # fit (fast)
        with st.spinner("Training model..."):
            try:
                pipeline.fit(X_train, y_train)
            except Exception as e:
                st.error(f"Training failed: {e}")
                return

        st.success("Model trained on train split — evaluating on test split.")

        # predict & show metrics
        try:
            y_pred = pipeline.predict(X_test)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        if task == "classification":
            y_prob = None
            try:
                if hasattr(pipeline, "predict_proba"):
                    y_prob = pipeline.predict_proba(X_test)
                elif hasattr(pipeline.named_steps["model"], "predict_proba"):
                    y_prob = pipeline.named_steps["model"].predict_proba(pipeline.named_steps["preprocessor"].transform(X_test))
            except Exception:
                y_prob = None

            metrics = _classification_metrics(y_test, y_pred, y_prob=y_prob)
            st.subheader("Classification metrics (test)")
            st.json(metrics)

            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Pred"); ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification report")
            st.text(classification_report(y_test, y_pred, zero_division=0))

        else:
            metrics = _regression_metrics(y_test, y_pred)
            st.subheader("Regression metrics (test)")
            st.json(metrics)
            fig, ax = plt.subplots(figsize=(5,3))
            ax.scatter(y_pred, np.asarray(y_test, dtype=float) - np.asarray(y_pred, dtype=float), alpha=0.6)
            ax.axhline(0, color="red", ls="--")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
            st.pyplot(fig)

        # optional CV (small folds)
        if do_cv:
            st.info("Running short CV (3 folds) — this may take time.")
            try:
                scoring = "accuracy" if task=="classification" else "r2"
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
                st.write(f"CV ({scoring}) mean = {cv_scores.mean():.4f}, std = {cv_scores.std():.4f}")
            except Exception as e:
                st.info(f"CV failed: {e}")

        # try to show simple feature importance if available
        try:
            model_obj = pipeline.named_steps["model"]
            if hasattr(model_obj, "feature_importances_"):
                # need feature names after preprocessor; best-effort: numeric + placeholder one-hot names
                pre = pipeline.named_steps["preprocessor"]
                feat_names = []
                # numeric list
                try:
                    feat_names += pre.transformers_[0][2]
                except Exception:
                    pass
                if hasattr(model_obj, "feature_importances_"):
                    fi = model_obj.feature_importances_
                    # show top raw values if names missing
                    if len(feat_names) == len(fi):
                        fi_s = pd.Series(fi, index=feat_names).sort_values(ascending=False)
                        st.dataframe(fi_s.head(20).to_frame("importance"))
                    else:
                        top_idx = np.argsort(fi)[-10:][::-1]
                        top_vals = {f"f{i}": float(fi[i]) for i in top_idx}
                        st.write("Top feature importance (raw indices):", top_vals)
            elif hasattr(model_obj, "coef_"):
                coef = np.abs(model_obj.coef_).ravel()
                st.write("Model coefficients (abs):", coef[:20].tolist())
            else:
                st.info("No feature importance available for this model type.")
        except Exception:
            pass

        # save pipeline to session
        st.session_state[SESSION_MODEL] = pipeline
        st.success("Trained pipeline saved to session for prediction/export.")

        # Predict UI (simple)
        st.markdown("### Quick predict (single-row)")
        if st.checkbox("Show simple predict UI"):
            st.info("Provide values for features (leave blank → treated as missing and imputed).")
            values = {}
            cols = st.columns(3)
            for i, f in enumerate(features):
                col = cols[i % 3]
                if pd.api.types.is_numeric_dtype(df[f]):
                    val = col.text_input(f"{f} (numeric)", value="")
                    values[f] = None if val=="" else float(val)
                else:
                    val = col.text_input(f"{f} (text/cat)", value="")
                    values[f] = None if val=="" else val
            if st.button("Predict this row"):
                try:
                    sample = pd.DataFrame([values])
                    pred = pipeline.predict(sample)[0]
                    st.metric("Prediction", str(pred))
                    if hasattr(pipeline, "predict_proba"):
                        try:
                            proba = pipeline.predict_proba(sample)[0]
                            st.write("Probabilities:", proba.tolist())
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Predict failed: {e}")

        # SHAP (optional, compact)
        if enable_shap and SHAP_AVAILABLE:
            st.markdown("### SHAP explanation (compact)")
            st.info("SHAP may be slow. Using a small background sample.")
            try:
                df_features = working[features].dropna(axis=0, how="all")
                bg = df_features.sample(min(100, len(df_features)), random_state=0)
                expl = df_features.sample(min(100, len(df_features)), random_state=1)
                pre = pipeline.named_steps["preprocessor"]
                model_obj = pipeline.named_steps["model"]
                Xb = pre.transform(bg)
                Xe = pre.transform(expl)
                # prefer TreeExplainer when applicable
                if hasattr(model_obj, "feature_importances_") or "forest" in model_obj.__class__.__name__.lower() or "xgb" in model_obj.__class__.__name__.lower():
                    explainer = shap.TreeExplainer(model_obj, data=Xb, feature_perturbation="tree_path_dependent")
                    shap_vals = explainer.shap_values(Xe)
                else:
                    explainer = shap.Explainer(model_obj, Xb)
                    shap_res = explainer(Xe)
                    shap_vals = shap_res.values
                # compute mean abs
                if isinstance(shap_vals, list):
                    abs_means = np.mean([np.mean(np.abs(s), axis=0) for s in shap_vals], axis=0)
                else:
                    abs_means = np.mean(np.abs(shap_vals), axis=0)
                # feature names best-effort:
                feat_names = []
                try:
                    feat_names += pre.transformers_[0][2]
                except Exception:
                    feat_names = [f"f{i}" for i in range(len(abs_means))]
                mean_abs = pd.Series(abs_means, index=feat_names).sort_values(ascending=False).head(20)
                st.dataframe(mean_abs.to_frame("mean_abs_shap"))
                fig, ax = plt.subplots(figsize=(6, max(3, 0.25*len(mean_abs))))
                mean_abs.sort_values().plot(kind="barh", ax=ax)
                ax.set_xlabel("Mean |SHAP|")
                st.pyplot(fig)
            except Exception as e:
                st.info(f"SHAP explanation failed or is unavailable: {e}")
        elif enable_shap and not SHAP_AVAILABLE:
            st.info("SHAP library not installed. Install with `pip install shap` to enable SHAP explanations.")

    # show model from session if present and not just trained
    st.markdown("---")
    st.subheader("Saved model in session")
    if SESSION_MODEL in st.session_state:
        st.success("A trained pipeline is present in session.")
        if st.button("Download trained pipeline (pickle)"):
            try:
                buf = io.BytesIO()
                pickle.dump(st.session_state[SESSION_MODEL], buf)
                buf.seek(0)
                st.download_button("Download model", data=buf.getvalue(), file_name="trained_pipeline.pkl")
            except Exception as e:
                st.info(f"Download failed: {e}")
        if st.button("Show brief summary"):
            st.json({"model": str(st.session_state[SESSION_MODEL].named_steps["model"].__class__.__name__)})
    else:
        st.info("No trained model in session yet. Train a model above to enable prediction/export.")
