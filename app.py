import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="E-commerce Customer Segmentation", page_icon="🛒", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

DEFAULT_SEGMENT_NAMES = {
    0: "Regular Customers",
    1: "High-Value Customers",
    2: "Inactive Customers",
}


@st.cache_resource
def load_artifacts():
    """Load saved models and metadata."""
    kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    future_model = joblib.load(MODEL_DIR / "future_purchase_model.pkl")

    selected_features_path = MODEL_DIR / "selected_features.pkl"
    segment_names_path = MODEL_DIR / "segment_names.pkl"

    selected_features = None
    segment_names = DEFAULT_SEGMENT_NAMES.copy()

    if selected_features_path.exists():
        selected_features = joblib.load(selected_features_path)

    if segment_names_path.exists():
        loaded_segment_names = joblib.load(segment_names_path)
        if isinstance(loaded_segment_names, dict):
            segment_names = loaded_segment_names

    return kmeans, scaler, future_model, selected_features, segment_names


def get_segment_strategy(segment_name: str):
    """Return business recommendation for each segment."""
    strategies = {
        "High-Value Customers": {
            "summary": "These customers buy frequently and spend more. They are the most valuable group.",
            "actions": [
                "Offer loyalty rewards and VIP benefits",
                "Recommend premium or bundle products",
                "Provide early access to sales and new arrivals",
            ],
        },
        "Regular Customers": {
            "summary": "These customers purchase consistently but are not yet top-value customers.",
            "actions": [
                "Use upselling and cross-selling campaigns",
                "Send personalized product recommendations",
                "Offer limited-time discounts to increase basket size",
            ],
        },
        "Inactive Customers": {
            "summary": "These customers have low recent activity and may be at risk of churn.",
            "actions": [
                "Run win-back email campaigns",
                "Offer reactivation discounts or coupons",
                "Send reminders based on previous purchase history",
            ],
        },
    }

    return strategies.get(
        segment_name,
        {
            "summary": "This segment was identified by the clustering model.",
            "actions": [
                "Review customer behavior carefully",
                "Use targeted marketing based on purchase patterns",
                "Monitor retention and purchase activity",
            ],
        },
    )


def prepare_segment_input(recency, frequency, monetary):
    return pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"]
    )


def prepare_future_input(recency, frequency, monetary, total_items, avg_order_value, selected_features):
    full_feature_map = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "TotalItems": total_items,
        "AvgOrderValue": avg_order_value,
    }

    if selected_features is None:
        selected_features = ["Recency", "Frequency", "Monetary", "TotalItems", "AvgOrderValue"]

    values = [full_feature_map[col] for col in selected_features]
    return pd.DataFrame([values], columns=selected_features)


def predict_future_label(pred):
    if int(pred) == 1:
        return "Likely to Purchase Again"
    return "Less Likely to Purchase Again"


# -----------------------------
# App
# -----------------------------
st.title("🛒 E-commerce Customer Segmentation and Prediction")
st.markdown(
    "This app predicts a customer's **segment** using the K-Means clustering model "
    "and estimates whether the customer is **likely to purchase again** using the saved classifier."
)

with st.sidebar:
    st.header("Model Files")
    st.write("Expected folder structure:")
    st.code(
        "project_folder/\n"
        "├─ app.py\n"
        "└─ saved_models/\n"
        "   ├─ kmeans_model.pkl\n"
        "   ├─ scaler.pkl\n"
        "   ├─ future_purchase_model.pkl\n"
        "   ├─ selected_features.pkl\n"
        "   └─ segment_names.pkl",
        language="text",
    )

    st.info("Run your notebook first so the .pkl files are created inside the saved_models folder.")

try:
    kmeans, scaler, future_model, selected_features, segment_names = load_artifacts()
except Exception as e:
    st.error("Could not load model files.")
    st.exception(e)
    st.stop()

tab1, tab2 = st.tabs(["Single Customer Prediction", "Batch Prediction (CSV)"])


# -----------------------------
# Tab 1: Single Prediction
# -----------------------------
with tab1:
    st.subheader("Predict for One Customer")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency", min_value=0.0, value=30.0, step=1.0,
                                  help="Days since the customer's last purchase")
    with col2:
        frequency = st.number_input("Frequency", min_value=0.0, value=5.0, step=1.0,
                                    help="Number of purchases")
    with col3:
        monetary = st.number_input("Monetary", min_value=0.0, value=500.0, step=10.0,
                                   help="Total spending")

    col4, col5 = st.columns(2)
    with col4:
        total_items = st.number_input("TotalItems", min_value=0.0, value=20.0, step=1.0,
                                      help="Total number of items purchased")
    with col5:
        avg_order_value = st.number_input("AvgOrderValue", min_value=0.0, value=100.0, step=1.0,
                                          help="Average value per order")

    if st.button("Predict Customer", type="primary"):
        # Segment prediction
        segment_input = prepare_segment_input(recency, frequency, monetary)
        scaled_segment_input = scaler.transform(segment_input)
        cluster = int(kmeans.predict(scaled_segment_input)[0])
        segment_name = segment_names.get(cluster, f"Segment {cluster}")

        # Future purchase prediction
        future_input = prepare_future_input(
            recency, frequency, monetary, total_items, avg_order_value, selected_features
        )
        future_pred = int(future_model.predict(future_input)[0])
        future_label = predict_future_label(future_pred)

        left, right = st.columns(2)

        with left:
            st.success(f"Predicted Segment: **{segment_name}**")
            st.metric("Cluster ID", cluster)

            strategy = get_segment_strategy(segment_name)
            st.markdown("**Segment Summary**")
            st.write(strategy["summary"])

            st.markdown("**Recommended Actions**")
            for action in strategy["actions"]:
                st.write(f"- {action}")

        with right:
            st.info(f"Future Purchase Prediction: **{future_label}**")

            if hasattr(future_model, "predict_proba"):
                probability = float(future_model.predict_proba(future_input)[0][1])
                st.metric("Probability of Future Purchase", f"{probability:.2%}")

        st.markdown("### Input Used")
        st.dataframe(
            pd.DataFrame(
                {
                    "Recency": [recency],
                    "Frequency": [frequency],
                    "Monetary": [monetary],
                    "TotalItems": [total_items],
                    "AvgOrderValue": [avg_order_value],
                }
            ),
            use_container_width=True,
        )


# -----------------------------
# Tab 2: Batch Prediction
# -----------------------------
with tab2:
    st.subheader("Batch Prediction from CSV")
    st.write(
        "Upload a CSV file containing these columns: "
        "`Recency`, `Frequency`, `Monetary`, `TotalItems`, `AvgOrderValue`."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("**Preview of uploaded data**")
        st.dataframe(df.head(), use_container_width=True)

        required_columns = ["Recency", "Frequency", "Monetary", "TotalItems", "AvgOrderValue"]
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            # Segment prediction
            segment_df = df[["Recency", "Frequency", "Monetary"]].copy()
            segment_scaled = scaler.transform(segment_df)
            clusters = kmeans.predict(segment_scaled)

            # Future purchase prediction
            if selected_features is None:
                selected_features = required_columns.copy()

            future_df = df[selected_features].copy()
            future_preds = future_model.predict(future_df)

            result_df = df.copy()
            result_df["Cluster"] = clusters
            result_df["SegmentName"] = result_df["Cluster"].map(
                lambda x: segment_names.get(int(x), f"Segment {int(x)}")
            )
            result_df["FuturePurchasePrediction"] = [
                predict_future_label(x) for x in future_preds
            ]

            if hasattr(future_model, "predict_proba"):
                result_df["FuturePurchaseProbability"] = future_model.predict_proba(future_df)[:, 1]

            st.markdown("### Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                data=csv,
                file_name="customer_predictions.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Built with Streamlit using your saved notebook models.")
