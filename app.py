# app.py — main Streamlit entrypoint (updated to call data_cleaning.render_data_cleaning)
import streamlit as st
from data_upload import render_data_upload
from data_cleaning import render_data_cleaning
from eda import render_eda
from automl import render_automl
from about import render_about
from nl2sql import render_nl2sql
# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="AutoAnalyst AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# theme / inline CSS (kept same as before)
THEME = {
    "primary": "#2563EB",
    "accent": "#06B6D4",
    "bg": "#0f172a",
    "panel": "rgba(255,255,255,0.02)",
    "text": "#E6EEF8",
    "muted": "#9CA3AF",
    "radius": "12px",
}

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {THEME['bg']} 0%, #071029 100%);
        color: {THEME['text']};
        font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .app-header {{ padding:14px; border-radius:12px; background: rgba(255,255,255,0.01); margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; }}
    .app-title {{ margin:0; font-size:20px; font-weight:700; }}
    .app-subtitle {{ margin:0; font-size:13px; color:{THEME['muted']}; margin-top:6px; }}
    .card {{ background: linear-gradient(180deg, rgba(255,255,255,0.012), rgba(255,255,255,0.006)); padding:18px; border-radius:12px; box-shadow: 0 8px 30px rgba(2,6,23,0.45); border: 1px solid rgba(255,255,255,0.02); }}
    section[data-testid="stSidebar"] > div:first-child {{ padding: 16px; margin: 8px; border-radius: 14px; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.007)); border: 1px solid rgba(255,255,255,0.02); box-shadow: 0 10px 30px rgba(2,6,23,0.35); }}
    [data-testid="stSidebar"] .stRadio > div label {{ display:flex; align-items:center; gap:10px; width:100%; padding:10px 12px; margin-bottom:8px; border-radius:10px; cursor:pointer; color:{THEME['text']}; background: transparent; border: 1px solid rgba(255,255,255,0.03); transition: transform 0.12s ease, box-shadow 0.12s ease; }}
    [data-testid="stSidebar"] input[type="radio"] {{ display:none; }}
    [data-testid="stSidebar"] .stRadio > div label[aria-checked="true"] {{ background: linear-gradient(90deg, {THEME['primary']}, {THEME['accent']}); color: white !important; border: none; transform: translateY(-2px); box-shadow: 0 12px 30px rgba(2,6,23,0.45); }}
    .feature-tile {{ padding:14px; border-radius:12px; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.02); }}
    .feature-icon {{ width:46px; height:46px; border-radius:10px; display:flex; align-items:center; justify-content:center; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); font-size:20px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="app-header">
      <div style="display:flex;flex-direction:column;">
        <div style="display:flex;align-items:baseline;gap:12px;">
          <h1 class="app-title">AutoAnalyst AI</h1>
          <div style="font-size:12px;color: #9CA3AF; background: rgba(255,255,255,0.02); padding:6px 10px; border-radius:10px;">v0.1</div>
        </div>
        <p class="app-subtitle">📤 Upload • 🧹 Clean • 📊 Explore • 🖼️ Visualize</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <div style="font-size:13px;color: #9CA3AF;">Built for fast analysis</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.markdown("")
st.sidebar.markdown(
    """
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:12px;">
      <div style="width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;background: linear-gradient(180deg, #2563EB, #06B6D4);color:white;font-weight:700;">AI</div>
      <div style="display:flex;flex-direction:column;">
        <div style="font-size:12px;color: #9CA3AF;">Project</div>
        <div style="font-size:13px;color: #E6EEF8;font-weight:600;">AutoAnalyst AI</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

menu_items = [
    ("🏠 Home", "Home"),
    ("📥 Data Upload", "Data Upload"),
    ("🧹 Data Cleaning", "Data Cleaning"),
    ("📈 EDA", "EDA"),
    ("🤖 AutoML", "AutoML"),
    ("🗣️ NL → SQL", "NL → SQL"),
    ("ℹ️ About", "About"),
]
labels = [item[0] for item in menu_items]
page_label = st.sidebar.radio("", labels, index=0, key="ui_menu")
label_to_page = {item[0]: item[1] for item in menu_items}
page = label_to_page.get(page_label, "Home")

st.sidebar.markdown('<div style="margin-top:8px;"><button style="width:100%;padding:10px 12px;border-radius:10px;border:none;background:linear-gradient(90deg,#2563EB,#06B6D4);color:white;font-weight:700">+ New analysis</button></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="margin-top:12px;font-size:12px;color:#9CA3AF;text-align:center">Made with ❤️ — UI Skeleton</div>', unsafe_allow_html=True)

# Helper renderers used on Home (same as before)
def feature_tile(icon, title, desc):
    st.markdown(
        f"""
        <div class="feature-tile" style="display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;">
          <div class="feature-icon">{icon}</div>
          <div style="flex:1;">
            <div style="font-weight:700;color:{THEME['text']};margin-bottom:6px">{title}</div>
            <div style="color:{THEME['muted']};font-size:13px">{desc}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def card(title, lines):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(title)
    for l in lines:
        st.write(l)
    st.markdown('</div>', unsafe_allow_html=True)

# Page routing
if page == "Home":
    # Full Home page as previously designed (keeps same look)
    left, right = st.columns([2.5, 1], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Welcome to **AutoAnalyst AI**")
        st.write(
            "A fast, focused UI scaffold to upload datasets, run smart cleaning, and generate exploratory visuals — all inside your Codespace. "
            "This Home page is a launchpad: quick start steps, handy feature tiles, and one-click actions are below."
        )
        st.write("")  # spacer

        # Feature row (Upload, Clean, Explore, Visualize) with icons
        st.markdown("#### Core workflow")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            feature_tile("📥", "Upload", "Drop CSV/Excel files or connect to a database to ingest data quickly.")
            st.write("")  # small spacer
            feature_tile("🧹", "Clean", "Smart imputations, type conversions, and quick de-duplication controls.")
        with col2:
            feature_tile("📊", "Explore", "Automatic summary stats, correlation matrix, and column-wise overviews.")
            st.write("")  # small spacer
            feature_tile("🖼️", "Visualize", "Interactive charts — histograms, scatter, line, and dashboard-ready visuals.")

        st.markdown("---")

        st.markdown("#### Getting started")
        st.markdown(
            """
            1. Go to **Data Upload** and add a dataset (CSV / Excel).  
            2. Visit **Data Cleaning** and run the quick-clean to fill missing values.  
            3. Open **EDA** to examine summary stats and interactive plots.  
            4. If you want SQL from plain English, try **NL → SQL** (prototype).  
            5. Use **AutoML** for a light demo of model training (placeholder).
            """
        )

        st.markdown("---")
        st.markdown("#### Quick actions")
        qa1, qa2, qa3 = st.columns([1,1,1], gap="small")
        with qa1:
            if st.button("📥 Upload sample CSV"):
                st.info("Sample CSV upload (UI-only placeholder).")
        with qa2:
            if st.button("📊 Open sample dashboard"):
                st.info("Sample dashboard would open (placeholder).")
        with qa3:
            if st.button("📁 View recent analyses"):
                st.info("Recent analyses list (UI-only).")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        card("Quick stats", [
            "Datasets: 0  (UI placeholder)",
            "Cleaned datasets: 0  (UI placeholder)",
            "Last run: —"
        ])
        st.markdown("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recent analyses")
        st.write("No recent analyses — this is a placeholder.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Need help?")
        st.write("• Read the README in the repo for setup instructions.")
        st.markdown('</div>', unsafe_allow_html=True)

# Data Upload
elif page == "Data Upload":
    render_data_upload()

# Data Cleaning (calls new module)
elif page == "Data Cleaning":
    render_data_cleaning()

# Other placeholders
elif page == "EDA":
    render_eda()

elif page == "AutoML":
    render_automl()

elif page == "NL → SQL":
    render_nl2sql()

elif page == "About":
    render_about()