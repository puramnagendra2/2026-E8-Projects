import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Cyber Attack Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ======================================
# LABEL MAPPING (CHANGE THESE NAMES AS NEEDED)
# ======================================
# Update these strings to match your dataset's actual attack names
LABEL_MAP = {
    0: "Malware",
    1: "DDoS",
    2: "Intrusion"
}

# ======================================
# SIDEBAR
# ======================================
with st.sidebar:
    st.title("🛡️ Cyber Dashboard")
    st.markdown("""
    - Upload dataset
    - View analytics
    - Predict attacks
    - View model evaluation
    """)
    st.info("AI-based Cyber Attack Detection")

    uploaded_file = st.file_uploader(
        "📂 Upload CSV File",
        type=["csv"]
    )

# ======================================
# HEADER
# ======================================
st.markdown(
    """
    <div style="text-align:center;">
        <h1>Cyber Attack Detection Dashboard</h1>
        <p>Analyze network data and detect security threats</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ======================================
# LOAD MODEL + METRICS
# ======================================
try:
    model = joblib.load("attack_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    encoder = joblib.load("label_encoder.pkl")
    model_metrics = joblib.load("model_metrics.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ======================================
# HELPER FUNCTION FOR READABLE LABELS
# ======================================
def make_readable(val):
    """Converts numeric predictions/labels to text using Map or Encoder."""
    try:
        # Try manual map first
        if int(val) in LABEL_MAP:
            return LABEL_MAP[int(val)]
        # Try inverse transform from encoder
        return encoder.inverse_transform([int(val)])[0]
    except:
        return str(val)

# ======================================
# PROCESS UPLOADED FILE
# ======================================
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Convert existing 'Attack Type' column to readable names if it exists
    if "Attack Type" in data.columns:
        data["Attack Type"] = data["Attack Type"].apply(make_readable)

    st.subheader("📊 Uploaded Dataset Overview")
    colA, colB, colC = st.columns(3)
    colA.metric("Total Rows", len(data))
    colB.metric("Columns", len(data.columns))
    colC.metric("Missing Values", data.isnull().sum().sum())

    st.write("---")
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head(10), use_container_width=True)
    st.write("---")

    if st.button("🚀 Run Threat Analysis", use_container_width=True):
        with st.spinner("Running predictions..."):
            # Prepare Features
            X = data.drop(columns=["Attack Type"]) if "Attack Type" in data.columns else data.copy()
            X = pd.get_dummies(X)
            X = X.reindex(columns=model_columns, fill_value=0)

            # Predict
            raw_predictions = model.predict(X)
            
            # Map predictions to names
            data["Predicted_Attack_Type"] = [make_readable(p) for p in raw_predictions]

        st.success("✅ Prediction Completed")

        # ======================================
        # ROW-WISE RESULTS
        # ======================================
        st.subheader("🔍 Row-wise Prediction Results")
        if "Attack Type" in data.columns:
            st.dataframe(data[["Attack Type", "Predicted_Attack_Type"]], use_container_width=True)
        else:
            st.dataframe(data[["Predicted_Attack_Type"]], use_container_width=True)

        st.write("---")

        # ======================================
        # PREDICTED DISTRIBUTION (READABLE LABELS)
        # ======================================
        st.subheader("📈 Predicted Attack Distribution")

        counts = data["Predicted_Attack_Type"].value_counts().reset_index()
        counts.columns = ["Attack Type", "Count"]
        counts["Percentage (%)"] = (counts["Count"] / len(data) * 100).round(2)

        # Show Table
        st.dataframe(counts, use_container_width=True)

        # Show Chart
        fig = px.pie(
            counts, 
            names="Attack Type", 
            values="Count",
            color="Attack Type",
            hole=0.4,
            title="Distribution of Detected Threats"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("---")

        # ======================================
        # EVALUATION BLOCK
        # ======================================
        if "Attack Type" in data.columns:
            st.subheader("📊 Evaluation on Uploaded Dataset")

            y_true = data["Attack Type"].astype(str)
            y_pred = data["Predicted_Attack_Type"].astype(str)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

            m1, m2 = st.columns(2)
            m1.metric("Current Accuracy", f"{acc:.4f}")
            m2.metric("Current F1-Score", f"{f1:.4f}")

            st.subheader("Detailed Classification Report")
            report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

        # ======================================
        # DOWNLOAD RESULTS
        # ======================================
        st.download_button(
            "📥 Download Full Results",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="threat_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("📂 Upload a CSV file from the sidebar to begin analysis.")
