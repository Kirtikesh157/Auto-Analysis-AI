# automl.py
"""
Polished AutoML Streamlit page for AutoAnalyst AI.

Features:
- Detects problem type (classification/regression)
- Suggests candidate models and lets user pick & tune a few params
- Builds robust preprocessing pipeline (imputation + encoding + scaling)
- Trains the chosen model, shows metrics and CV
- Predict UI (single-record form + single-row CSV)
- SHAP model explanation: global mean(|SHAP|), beeswarm (optional), per-instance SHAP bar
- Saves model to session_state and supports model download
- Defensive, visually-appealing UI
"""

from typing import List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# sklearn imports (required)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, r2_score
)

# classifiers/regressors
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

# optional xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# optional shap
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# session keys
SESSION_RAW = "__uploaded_df__"
SESSION_CLEAN = "__cleaned_df__"
SESSION_MODEL = "__trained_model__"

# ---------------- Utility helpers ----------------
def _get_df_from_session() -> Tuple[pd.DataFrame, bool]:
    if SESSION_CLEAN in st.session_state:
        return st.session_state[SESSION_CLEAN], True
    if SESSION_RAW in st.session_state:
        return st.session_state[SESSION_RAW], False
    return None, False

def _auto_detect_task(df: pd.DataFrame, target_col: str) -> str:
    ser = df[target_col].dropna()
    if pd.api.types.is_numeric_dtype(ser):
        return "classification" if ser.nunique() <= 20 else "regression"
    return "classification"

def _get_candidate_models(task: str):
    models = {}
    if task == "classification":
        models["Logistic Regression"] = LogisticRegression
        models["Random Forest"] = RandomForestClassifier
        models["Gradient Boosting (sklearn)"] = GradientBoostingClassifier
        models["SVM (RBF)"] = SVC
        if XGBOOST_AVAILABLE:
            models["XGBoost (classifier)"] = xgb.XGBClassifier
    else:
        models["Linear Regression"] = LinearRegression
        models["Random Forest Regressor"] = RandomForestRegressor
        models["Gradient Boosting Regressor"] = GradientBoostingRegressor
        models["SVR (RBF)"] = SVR
        if XGBOOST_AVAILABLE:
            models["XGBoost (regressor)"] = xgb.XGBRegressor
    return models

def _onehot_encoder_compat():
    # create OneHotEncoder compatibly across sklearn versions
    from sklearn.preprocessing import OneHotEncoder as _OHE
    kwargs = {"handle_unknown": "ignore"}
    try:
        _ = _OHE(**{"sparse_output": False})
        kwargs["sparse_output"] = False
    except TypeError:
        try:
            _ = _OHE(**{"sparse": False})
            kwargs["sparse"] = False
        except TypeError:
            # very old sklearn, leave kwargs minimal
            pass
    except Exception:
        try:
            _ = _OHE(**{"sparse": False})
            kwargs["sparse"] = False
        except Exception:
            pass
    return _OHE, kwargs

def _build_preprocessor(df: pd.DataFrame, features: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    OHE, ohe_kwargs = _onehot_encoder_compat()
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OHE(**ohe_kwargs))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop")

    return preprocessor, num_cols, cat_cols

def _get_feature_names(preprocessor: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    names = []
    names += list(num_cols)
    try:
        # attempt to extract categories from fitted onehot encoder
        for name, trans, cols in preprocessor.transformers_:
            if name == "cat":
                # trans is a Pipeline
                if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
                    ohe = trans.named_steps["onehot"]
                    if hasattr(ohe, "categories_"):
                        for c, cats in zip(cols, ohe.categories_):
                            for cat in cats:
                                names.append(f"{c}__{cat}")
                    else:
                        # fallback: add col names without categories
                        for c in cols:
                            names.append(c)
    except Exception:
        # fallback: use numeric only or positional names
        pass
    return names

# metrics helpers
def _classification_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = float(roc_auc_score(y_true, y_prob[:,1] if y_prob.ndim>1 else y_prob))
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            metrics["roc_auc"] = auc
        except Exception:
            metrics["roc_auc"] = None
    return metrics

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

# ---------------- SHAP helpers ----------------
def _safe_shap_explain(pipeline: Pipeline, X_bg: pd.DataFrame, X_explain: pd.DataFrame, feature_names: List[str], max_display=20):
    """
    Safe SHAP explain: returns mean_abs Series and optionally raw shap values
    """
    if not SHAP_AVAILABLE:
        return None, "shap not installed"

    model = pipeline.named_steps.get("model", None)
    preprocessor = pipeline.named_steps.get("preprocessor", None)
    if model is None or preprocessor is None:
        return None, "pipeline missing components"

    try:
        Xb = preprocessor.transform(X_bg)
        Xe = preprocessor.transform(X_explain)
    except Exception as e:
        return None, f"preprocessor.transform failed: {e}"

    try:
        # For tree models prefer TreeExplainer
        model_name = model.__class__.__name__.lower()
        if ("forest" in model_name) or ("xgb" in model_name) or ("gradient" in model_name) or hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model, data=Xb, feature_perturbation="tree_path_dependent")
            shap_vals = explainer.shap_values(Xe)
        else:
            explainer = shap.Explainer(model, Xb)
            shap_res = explainer(Xe)  # shap.Explanation object
            shap_vals = shap_res.values
    except Exception as e:
        try:
            explainer = shap.Explainer(model, Xb)
            shap_res = explainer(Xe)
            shap_vals = shap_res.values
        except Exception as e2:
            return None, f"shap explainer failed: {e2}"

    try:
        if isinstance(shap_vals, list):
            abs_means = np.mean([np.mean(np.abs(s), axis=0) for s in shap_vals], axis=0)
        else:
            abs_means = np.mean(np.abs(shap_vals), axis=0)
        mean_abs_series = pd.Series(abs_means, index=feature_names).sort_values(ascending=False)
        return {"shap_values": shap_vals, "mean_abs": mean_abs_series.head(max_display)}, None
    except Exception as e:
        return None, f"shap postprocessing failed: {e}"

# ---------------- UI / Render ----------------
def render_automl():
    st.set_page_config(page_title="AutoAnalyst — AutoML", layout="wide")
    st.title("AutoML — Train · Predict · Explain")
    st.caption("Suggests sensible models, trains chosen model, predicts single records and explains with SHAP (if available).")

    df, used_clean = _get_df_from_session()
    if df is None:
        st.warning("No dataset in session. Upload data on the Data Upload page first.")
        return

    # Top-row metrics
    st.header("Dataset overview")
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    missing_pct = 1 - (df.notna().sum().sum() / (df.shape[0] * df.shape[1]))
    col3.metric("Missing %", f"{missing_pct*100:.2f}%")
    unique_pct = df.nunique().mean() / df.shape[0] if df.shape[0] else 0
    col4.metric("Avg unique ratio", f"{unique_pct:.2f}")

    st.markdown("---")

    # 1) Target selection & task detection
    st.subheader("1) Choose target & task")
    target_col = st.selectbox("Select the target column (what you want to predict)", options=df.columns.tolist())
    if target_col is None:
        return

    st.write("Target preview:")
    if pd.api.types.is_numeric_dtype(df[target_col]):
        st.table(df[target_col].describe().to_frame().T)
    else:
        st.table(df[target_col].value_counts().head(10).to_frame("count"))

    detected_task = _auto_detect_task(df, target_col)
    task = st.radio("Detected task — override if needed", options=["classification", "regression"], index=0 if detected_task=="classification" else 1)
    st.info(f"Detected: {detected_task}. Using: {task}")

    working_df = df.dropna(subset=[target_col]).copy()
    if working_df.shape[0] < df.shape[0]:
        st.info(f"Dropped {df.shape[0] - working_df.shape[0]} rows with missing target for training.")

    # 2) Feature selection
    st.subheader("2) Feature selection")
    feature_candidates = [c for c in working_df.columns if c != target_col]
    use_all = st.checkbox("Use all other columns (recommended)", value=True)
    if not use_all:
        features = st.multiselect("Select feature columns", options=feature_candidates, default=feature_candidates[:min(6, len(feature_candidates))])
    else:
        features = feature_candidates

    if not features:
        st.error("No features selected.")
        return

    # 3) Model selection
    st.subheader("3) Choose model & hyperparameters")
    candidates = _get_candidate_models(task)
    st.write("Suggested models:")
    st.write(", ".join(list(candidates.keys())))
    model_choice = st.selectbox("Pick a model to train", options=list(candidates.keys()), index=0)

    # Simple hyperparameter controls
    params = {}
    if model_choice in ["Random Forest", "Random Forest Regressor"]:
        n_estimators = st.number_input("n_estimators", min_value=10, max_value=2000, value=100, step=10)
        max_depth = st.number_input("max_depth (0 = None)", min_value=0, max_value=100, value=0, step=1)
        params["n_estimators"] = int(n_estimators)
        params["max_depth"] = None if int(max_depth) == 0 else int(max_depth)
    elif model_choice.startswith("Gradient Boosting"):
        n_estimators = st.number_input("n_estimators", min_value=10, max_value=2000, value=100, step=10)
        learning_rate = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
        params["n_estimators"] = int(n_estimators)
        params["learning_rate"] = float(learning_rate)
    elif model_choice == "Logistic Regression":
        C = st.number_input("C (inverse regularization)", min_value=1e-4, max_value=1e4, value=1.0, format="%.4f")
        max_iter = st.number_input("max_iter", min_value=50, max_value=2000, value=200, step=50)
        params["C"] = float(C); params["max_iter"] = int(max_iter)
    elif model_choice in ["SVM (RBF)", "SVR (RBF)"]:
        C = st.number_input("C", min_value=0.01, max_value=1000.0, value=1.0, format="%.2f")
        params["C"] = float(C)
        if model_choice == "SVM (RBF)":
            params["probability"] = True
    elif model_choice.startswith("XGBoost"):
        n_estimators = st.number_input("n_estimators", min_value=10, max_value=2000, value=100, step=10)
        lr = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
        params["n_estimators"] = int(n_estimators); params["learning_rate"] = float(lr)

    # 4) Train/test split
    st.subheader("4) Train / Test split & training")
    test_size = st.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    random_state = int(st.number_input("Random seed", min_value=0, max_value=9999, value=42))

    # Train button (wide area)
    train_col1, train_col2 = st.columns([1, 3])
    with train_col1:
        train_button = st.button("Train model", key="train_button")

    if train_button:
        preprocessor, num_cols, cat_cols = _build_preprocessor(working_df, features)
        ModelClass = _get_candidate_models(task)[model_choice]
        try:
            model = ModelClass(**params) if params else ModelClass()
        except Exception as e:
            st.error(f"Model instantiation failed: {e}")
            return

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # stratify for classification if possible
        stratify = working_df[target_col] if (task == "classification" and working_df[target_col].nunique() > 1) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(working_df[features], working_df[target_col],
                                                                test_size=float(test_size), random_state=int(random_state),
                                                                stratify=stratify)
        except Exception as e:
            st.error(f"Train/test split error: {e}")
            return

        with st.spinner("Training model... this may take a moment"):
            try:
                pipeline.fit(X_train, y_train)
            except Exception as e:
                st.error(f"Training failed: {e}")
                return

        st.success("Model trained — evaluating on test set.")

        # predict & metrics
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
            st.subheader("Classification metrics (test set)")
            st.json(metrics)

            # confusion matrix & report
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification report")
            st.text(classification_report(y_test, y_pred, zero_division=0))

            # ROC if binary and probabilities present
            if y_prob is not None and len(np.unique(y_test)) == 2:
                try:
                    from sklearn.metrics import roc_curve, auc
                    prob_pos = y_prob[:,1]
                    fpr, tpr, _ = roc_curve(y_test, prob_pos)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                    ax.plot([0,1],[0,1], linestyle="--", color="gray")
                    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC curve")
                    ax.legend()
                    st.pyplot(fig)
                except Exception:
                    pass

        else:
            y_pred_arr = np.array(y_pred).astype(float)
            y_test_arr = np.array(y_test).astype(float)
            metrics = _regression_metrics(y_test_arr, y_pred_arr)
            st.subheader("Regression metrics (test set)")
            st.json(metrics)

            # residual plot
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(y_pred_arr, y_test_arr - y_pred_arr, alpha=0.6)
            ax.axhline(0, color="red", linestyle="--")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
            st.pyplot(fig)

        # cross-validation (5-fold)
        st.subheader("Cross-validation (5-fold)")
        try:
            scoring = "accuracy" if task == "classification" else "r2"
            cv_scores = cross_val_score(pipeline, working_df[features], working_df[target_col], cv=5, scoring=scoring, n_jobs=-1)
            st.write(f"CV ({scoring}): mean = {cv_scores.mean():.4f}, std = {cv_scores.std():.4f}")
        except Exception as e:
            st.info(f"Cross-validation skipped/failed: {e}")

        # feature importance / coefficients
        st.subheader("Feature importance / coefficients (if available)")
        try:
            # fit preprocessor already happened; extract feature names
            feat_names = _get_feature_names(pipeline.named_steps["preprocessor"], num_cols, cat_cols)
            model_obj = pipeline.named_steps["model"]
            fi = None
            if hasattr(model_obj, "feature_importances_"):
                fi = model_obj.feature_importances_
            elif hasattr(model_obj, "coef_"):
                fi = np.abs(model_obj.coef_).ravel()
            if fi is not None:
                if feat_names and len(feat_names) == len(fi):
                    fi_series = pd.Series(fi, index=feat_names).sort_values(ascending=False)
                    st.dataframe(fi_series.head(30).to_frame("importance"))
                    fig, ax = plt.subplots(figsize=(6, max(3, 0.25*min(20, len(fi_series)))))
                    fi_series.head(20).plot(kind="barh", ax=ax)
                    ax.invert_yaxis()
                    st.pyplot(fig)
                else:
                    # fallback: show top raw values
                    top_idx = np.argsort(fi)[-20:][::-1]
                    top_vals = {f"f_{i}": float(fi[i]) for i in top_idx}
                    st.write(top_vals)
            else:
                st.info("Model does not expose feature importances or coefficients.")
        except Exception as e:
            st.info(f"Feature importance extraction failed: {e}")

        # save pipeline to session
        st.session_state[SESSION_MODEL] = pipeline
        st.success("Trained pipeline saved to session.")

        # allow model download
        try:
            buf = io.BytesIO()
            pickle.dump(pipeline, buf)
            buf.seek(0)
            st.download_button("Download trained model (pickle)", data=buf.getvalue(), file_name=f"trained_model_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.pkl")
        except Exception as e:
            st.info(f"Could not create download: {e}")

        # store summary
        st.session_state["__automl_summary__"] = {"model_choice": model_choice, "params": params, "metrics": metrics, "task": task}

    # ------ Prediction UI -------
    st.markdown("---")
    st.subheader("Predict — Single record")
    if SESSION_MODEL in st.session_state:
        pipeline = st.session_state[SESSION_MODEL]
        model_obj = pipeline.named_steps["model"]

        col_a, col_b = st.columns([1,2])
        with col_a:
            use_form = st.checkbox("Use manual form input", value=False)
            upload_csv = st.file_uploader("Or upload a single-row CSV", type=["csv"], help="CSV must contain the same feature columns.")
            if use_form:
                st.info("Leave inputs blank to treat them as missing (will be imputed).")
                inputs = {}
                with st.form("single_form"):
                    for f in features:
                        if pd.api.types.is_numeric_dtype(df[f]):
                            val = st.text_input(f"{f} (numeric)", value="")
                            inputs[f] = None if val == "" else float(val)
                        else:
                            val = st.text_input(f"{f} (categorical)", value="")
                            inputs[f] = None if val == "" else val
                    submitted = st.form_submit_button("Predict")
                if submitted:
                    sample = pd.DataFrame([inputs])
                    try:
                        pred = pipeline.predict(sample)[0]
                        st.metric("Prediction", str(pred))
                        if hasattr(pipeline, "predict_proba") or hasattr(model_obj, "predict_proba"):
                            try:
                                proba = pipeline.predict_proba(sample)[0]
                                st.write("Probabilities:", proba.tolist())
                            except Exception:
                                st.info("Probability prediction not available.")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

            if upload_csv is not None:
                try:
                    df_pred = pd.read_csv(upload_csv)
                    if df_pred.shape[0] != 1:
                        st.error("CSV must have exactly one row.")
                    else:
                        try:
                            pred = pipeline.predict(df_pred)[0]
                            st.metric("Prediction (CSV)", str(pred))
                            if hasattr(pipeline, "predict_proba") or hasattr(model_obj, "predict_proba"):
                                try:
                                    proba = pipeline.predict_proba(df_pred)[0]
                                    st.write("Probabilities:", proba.tolist())
                                except Exception:
                                    st.info("Probability prediction not available.")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
    else:
        st.info("No trained model in session — train one above to enable prediction.")

    # ------ SHAP explanations -------
    st.markdown("---")
    st.subheader("Model Explainability — SHAP (optional)")
    if not SHAP_AVAILABLE:
        st.info("SHAP not installed. Install with `pip install shap` to enable explanations.")
    elif SESSION_MODEL not in st.session_state:
        st.info("No trained model in session — train one above to enable SHAP explanations.")
    else:
        pipeline = st.session_state[SESSION_MODEL]
        st.write("Choose explanation mode and sizes (smaller sizes are faster).")
        bg_size = st.slider("Background sample size (for SHAP)", min_value=50, max_value=500, value=100, step=50)
        explain_size = st.slider("Explain sample size (for summary/beeswarm)", min_value=20, max_value=200, value=100, step=20)
        show_beeswarm = st.checkbox("Show SHAP beeswarm (slower)", value=False)
        show_instance = st.checkbox("Show per-instance SHAP bar for a selected row", value=False)

        if st.button("Run SHAP explanations"):
            try:
                # prepare data
                df_features = working_df[features].dropna(axis=0, how="all")
                if df_features.shape[0] == 0:
                    st.error("No usable rows for SHAP (all rows empty for chosen features).")
                else:
                    bg = df_features.sample(min(bg_size, len(df_features)), random_state=0)
                    expl = df_features.sample(min(explain_size, len(df_features)), random_state=1)

                    # get feature names
                    pre = pipeline.named_steps["preprocessor"]
                    num_cols = pre.transformers_[0][2] if len(pre.transformers_)>0 else []
                    cat_cols = pre.transformers_[1][2] if len(pre.transformers_)>1 else []
                    feat_names = _get_feature_names(pre, num_cols, cat_cols)
                    if not feat_names:
                        # attempt transform shape
                        try:
                            transformed = pre.transform(df_features.head(1))
                            feat_names = [f"f{i}" for i in range(transformed.shape[1])]
                        except Exception:
                            feat_names = [f"f{i}" for i in range(max(1, df_features.shape[1]))]

                    st.info("Computing SHAP values — this may take time (depends on model & sample sizes).")
                    with st.spinner("Running SHAP..."):
                        res, err = _safe_shap_explain(pipeline, bg, expl, feat_names, max_display=50)
                    if err:
                        st.error(f"SHAP failed: {err}")
                    else:
                        mean_abs = res["mean_abs"]
                        st.subheader("Global feature importance (mean |SHAP|)")
                        st.dataframe(mean_abs.to_frame("mean_abs_shap"))
                        fig, ax = plt.subplots(figsize=(6, max(3, 0.25*len(mean_abs))))
                        mean_abs.sort_values().plot(kind="barh", ax=ax)
                        ax.set_xlabel("Mean |SHAP value|")
                        st.pyplot(fig)

                        # beeswarm (optional)
                        if show_beeswarm:
                            try:
                                st.subheader("SHAP beeswarm (summary)")
                                # compute SHAP explainer and values directly for beeswarm
                                model = pipeline.named_steps["model"]
                                preprocessor = pipeline.named_steps["preprocessor"]
                                Xbg = preprocessor.transform(bg)
                                Xexp = preprocessor.transform(expl)
                                if hasattr(model, "predict_proba") and (hasattr(model, "feature_importances_") or "forest" in model.__class__.__name__.lower() or "xgb" in model.__class__.__name__.lower()):
                                    explainer = shap.TreeExplainer(model, data=Xbg, feature_perturbation="tree_path_dependent")
                                    shap_values = explainer.shap_values(Xexp)
                                else:
                                    explainer = shap.Explainer(model, Xbg)
                                    shap_values = explainer(Xexp).values
                                # shap.summary_plot can handle shap_values and feature matrix
                                plt.figure(figsize=(8,6))
                                try:
                                    shap.summary_plot(shap_values, Xexp, feature_names=feat_names, show=False)
                                    st.pyplot(plt.gcf())
                                except Exception as e:
                                    st.info(f"Could not render beeswarm: {e}")
                            except Exception as e:
                                st.info(f"Beeswarm failed: {e}")

                        # per-instance bar (fast)
                        if show_instance:
                            try:
                                st.subheader("Per-instance SHAP (select a row index from the explained sample)")
                                idx = st.number_input("Row index (0..n-1)", min_value=0, max_value=max(0, expl.shape[0]-1), value=0)
                                chosen_row = expl.reset_index(drop=True).iloc[[idx]]
                                # recompute SHAP values for chosen_row
                                res_i, err_i = _safe_shap_explain(pipeline, bg, chosen_row, feat_names, max_display=200)
                                if err_i:
                                    st.info(f"Instance SHAP failed: {err_i}")
                                else:
                                    mean_abs_i = res_i["mean_abs"]
                                    st.write("Top feature contributions (mean abs for instance):")
                                    st.dataframe(mean_abs_i.to_frame("mean_abs_shap"))
                                    fig, ax = plt.subplots(figsize=(6, max(3, 0.25*len(mean_abs_i))))
                                    mean_abs_i.sort_values().plot(kind="barh", ax=ax)
                                    ax.set_xlabel("Mean |SHAP value| for instance")
                                    st.pyplot(fig)
                            except Exception as e:
                                st.info(f"Per-instance SHAP failed: {e}")

                        # allow download of SHAP values (optional)
                        if "shap_values" in res:
                            try:
                                # prepare CSV of mean_abs
                                csv_buf = io.StringIO()
                                mean_abs.to_frame("mean_abs").to_csv(csv_buf)
                                st.download_button("Download SHAP mean_abs CSV", csv_buf.getvalue().encode("utf-8"), file_name="shap_mean_abs.csv")
                            except Exception:
                                pass

            except Exception as e:
                st.error(f"SHAP pipeline error: {e}")

    # show automl summary if present
    st.markdown("---")
    st.subheader("Saved model & summary")
    if "__automl_summary__" in st.session_state:
        if st.button("Show automl summary"):
            st.json(st.session_state["__automl_summary__"])
    else:
        st.info("No AutoML summary found (train a model to populate summary).")
