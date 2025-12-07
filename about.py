# about.py
"""
Clean About page for AutoAnalyst AI with:
EDA · AutoML · NL→SQL
"""

import streamlit as st
from datetime import datetime

def _css():
    st.markdown(
        """
        <style>
        .aa-hero {
            background: linear-gradient(90deg, #0f172a 0%, #0ea5a4 50%, #3b82f6 100%);
            color: white;
            padding: 36px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(10,10,10,0.3);
            margin-bottom: 18px;
        }
        .aa-hero h1 {
            font-size: 36px; margin: 0 0 8px 0; font-weight:700;
            letter-spacing: -0.5px;
        }
        .aa-hero p {
            font-size: 16px; margin: 0; color: rgba(255,255,255,0.92);
        }

        .aa-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            padding: 22px;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(16,24,40,0.15);
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 16px;
        }

        .aa-feature-title {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 6px;
            color: #e2e8f0;
        }
        .aa-feature-body {
            font-size: 15px;
            line-height: 1.45;
            color: #cbd5e1;
        }

        .aa-badge {
            background: rgba(255,255,255,0.15);
            color: white;
            padding: 5px 12px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 8px;
            display: inline-block;
        }

        .muted {
            color: #94a3b8;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_about():
    st.set_page_config(page_title="About — AutoAnalyst AI", layout="wide")
    _css()

    st.markdown(
        """
        <div class="aa-hero">
            <span class="aa-badge">AutoAnalyst AI</span>
            <h1>Exploratory Analysis · AutoML · NL→SQL </h1>
            <p>A clean, end-to-end workspace that takes you from raw data to insights and models — fast, accurate, and explainable.</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="aa-card">
        <div class="aa-feature-body">
        <b style="color:#e2e8f0; font-size:16px;">What AutoAnalyst AI does</b>
        <br><br>
        • <b>Upload & ingest:</b> CSV, Excel or SQL.<br>
        • <b>Data cleaning:</b> simple, accurate cleaning (median impute / drop).<br>
        • <b>EDA:</b> elegant visuals, missingness maps, correlations, outliers, one-paragraph narrative.<br>
        • <b>Natural Language → SQL:</b> convert English instructions to clean, executable SQL.<br>
        • <b>AutoML:</b> select a model, train it in your dataset and view evaluation metrics.<br>
        • <b>Predict & Explain:</b> Run predictions on new data and visualize feature contributions for transparent, explainable results.<br>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="aa-card">
                <div class="aa-badge">Visual EDA</div>
                <div class="aa-feature-title">Analyst-quality summaries</div>
                <div class="aa-feature-body">
                    Automated charts and narratives that avoid misleading insights through conservative heuristics: ID-column detection, filtered correlations, and correct Pareto logic.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="aa-card">
                <div class="aa-badge">AutoML</div>
                <div class="aa-feature-title">Model selection & SHAP</div>
                <div class="aa-feature-body">
                    Train a chosen algorithm, tune hyperparameters, evaluate model performance, and inspect SHAP explanations to understand how each feature influences predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="aa-card">
                <div class="aa-badge">NL → SQL</div>
                <div class="aa-feature-title">Natural Language to SQL</div>
                <div class="aa-feature-body">
                    Translate English queries into accurate SQL, leveraging your dataset’s schema and safe rule-based parsing. Optionally execute the query in a sandbox for immediate feedback.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="muted">
        AutoAnalyst AI · Clean · Accurate · Explainable<br>
        Version: {datetime.utcnow().strftime('%Y-%m-%d')}
        </div>
        """,
        unsafe_allow_html=True
    )
