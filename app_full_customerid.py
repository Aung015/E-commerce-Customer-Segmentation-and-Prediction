from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="E-commerce Customer Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main {padding-top: 1.2rem;}
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .hero-card {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .hero-card h1 {margin: 0; font-size: 2rem;}
        .hero-card p {margin: 0.5rem 0 0 0; color: #e2e8f0;}
        .section-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 1rem 1rem 0.5rem 1rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }
        .small-note {color: #64748b; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

DEFAULT_SEGMENT_NAMES = {
    0: "Regular Customers",
    1: "High-Value Customers",
    2: "Inactive Customers",
}

REQUIRED_COLUMNS = [
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary",
    "TotalItems",
    "AvgOrderValue",
]

MODEL_FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalItems",
    "AvgOrderValue",
]


@st.cache_resource
def load_artifacts():
    kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    future_model = joblib.load(MODEL_DIR / "future_purchase_model.pkl")

    selected_features = None
    segment_names = DEFAULT_SEGMENT_NAMES.copy()

    selected_features_path = MODEL_DIR / "selected_features.pkl"
    segment_names_path = MODEL_DIR / "segment_names.pkl"

    if selected_features_path.exists():
        selected_features = joblib.load(selected_features_path)

    if segment_names_path.exists():
        loaded_segment_names = joblib.load(segment_names_path)
        if isinstance(loaded_segment_names, dict):
            segment_names = loaded_segment_names

    return kmeans, scaler, future_model, selected_features, segment_names


def get_segment_strategy(segment_name: str):
    strategies = {
        "High-Value Customers": {
            "summary": "These customers purchase frequently and contribute higher revenue. They are the most valuable audience for retention and premium campaigns.",
            "actions": [
                "Offer loyalty rewards and VIP benefits",
                "Promote premium bundles and exclusive launches",
                "Use personalised retention campaigns and early access offers",
            ],
        },
        "Regular Customers": {
            "summary": "These customers show stable purchasing behaviour and represent an important growth segment with upselling potential.",
            "actions": [
                "Apply cross-sell and upsell campaigns",
                "Send personalised recommendations based on prior purchases",
                "Use limited-time offers to increase order value",
            ],
        },
        "Inactive Customers": {
            "summary": "These customers have weaker recent engagement and are more likely to need reactivation support.",
            "actions": [
                "Run win-back email campaigns",
                "Provide targeted discounts or reactivation coupons",
                "Send reminders based on previously purchased items",
            ],
        },
    }
    return strategies.get(
        segment_name,
        {
            "summary": "This segment was identified from customer purchasing patterns.",
            "actions": [
                "Monitor behaviour over time",
                "Design segment-specific promotions",
                "Track retention and repeat purchase rates",
            ],
        },
    )


def prepare_segment_input(recency, frequency, monetary):
    return pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"],
    )


def prepare_future_input(recency, frequency, monetary, total_items, avg_order_value, selected_features):
    feature_map = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "TotalItems": total_items,
        "AvgOrderValue": avg_order_value,
    }

    if selected_features is None:
        selected_features = MODEL_FEATURE_COLUMNS.copy()

    return pd.DataFrame(
        [[feature_map[c] for c in selected_features]],
        columns=selected_features,
    )


def predict_future_label(pred):
    return "Likely to Purchase Again" if int(pred) == 1 else "Less Likely to Purchase Again"


def validate_uploaded_dataframe(df: pd.DataFrame):
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, missing_cols
    return True, []


def add_actual_comparison_columns(result_df: pd.DataFrame):
    if "FuturePurchase" in result_df.columns:
        actual_numeric = pd.to_numeric(result_df["FuturePurchase"], errors="coerce")
        result_df["ActualFuturePurchase"] = actual_numeric.map(
            lambda x: "Likely to Purchase Again" if x == 1 else (
                "Less Likely to Purchase Again" if x == 0 else "Unknown"
            )
        )
        valid_mask = actual_numeric.isin([0, 1])
        if valid_mask.any():
            predicted_binary = result_df.loc[valid_mask, "FuturePurchasePrediction"].map(
                {"Likely to Purchase Again": 1, "Less Likely to Purchase Again": 0}
            )
            accuracy = (predicted_binary == actual_numeric.loc[valid_mask]).mean()
            return result_df, accuracy
    return result_df, None


def run_batch_predictions(df, scaler, kmeans, future_model, selected_features, segment_names):
    segment_df = df[["Recency", "Frequency", "Monetary"]].copy()
    segment_scaled = scaler.transform(segment_df)
    clusters = kmeans.predict(segment_scaled)

    if selected_features is None:
        selected_features = MODEL_FEATURE_COLUMNS.copy()

    future_df = df[selected_features].copy()
    future_preds = future_model.predict(future_df)

    result_df = df.copy()
    result_df["Cluster"] = clusters
    result_df["SegmentName"] = result_df["Cluster"].map(
        lambda x: segment_names.get(int(x), f"Segment {int(x)}")
    )
    result_df["FuturePurchasePrediction"] = [predict_future_label(x) for x in future_preds]

    if hasattr(future_model, "predict_proba"):
        result_df["FuturePurchaseProbability"] = future_model.predict_proba(future_df)[:, 1]

    result_df, uploaded_accuracy = add_actual_comparison_columns(result_df)

    preferred_order = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AvgOrderValue",
        "FuturePurchase",
        "ActualFuturePurchase",
        "Cluster",
        "SegmentName",
        "FuturePurchasePrediction",
        "FuturePurchaseProbability",
    ]
    ordered_cols = [c for c in preferred_order if c in result_df.columns]
    remaining_cols = [c for c in result_df.columns if c not in ordered_cols]
    result_df = result_df[ordered_cols + remaining_cols]

    return result_df, uploaded_accuracy


def render_kpi_row(result_df):
    total_customers = len(result_df)
    high_value = int((result_df["SegmentName"] == "High-Value Customers").sum())
    inactive = int((result_df["SegmentName"] == "Inactive Customers").sum())
    likely_purchase = int((result_df["FuturePurchasePrediction"] == "Likely to Purchase Again").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers Analysed", f"{total_customers:,}")
    c2.metric("High-Value Customers", f"{high_value:,}", f"{(high_value / total_customers * 100):.1f}%")
    c3.metric("Inactive Customers", f"{inactive:,}", f"{(inactive / total_customers * 100):.1f}%")
    c4.metric("Likely Repeat Buyers", f"{likely_purchase:,}", f"{(likely_purchase / total_customers * 100):.1f}%")


def render_segment_analytics(result_df):
    st.markdown("### Segment Analytics")

    left, right = st.columns(2)

    with left:
        segment_counts = result_df["SegmentName"].value_counts().rename_axis("Segment").reset_index(name="Count")
        st.markdown("**Customer Segment Distribution**")
        st.bar_chart(segment_counts.set_index("Segment"))

        segment_profile = (
            result_df.groupby("SegmentName")[["Recency", "Frequency", "Monetary", "AvgOrderValue"]]
            .mean()
            .round(2)
            .sort_values("Monetary", ascending=False)
        )
        st.markdown("**Average Segment Profile**")
        st.dataframe(segment_profile, use_container_width=True)

    with right:
        future_counts = (
            result_df["FuturePurchasePrediction"]
            .value_counts()
            .rename_axis("Prediction")
            .reset_index(name="Count")
        )
        st.markdown("**Future Purchase Outlook**")
        st.bar_chart(future_counts.set_index("Prediction"))

        segment_revenue = (
            result_df.groupby("SegmentName")["Monetary"]
            .sum()
            .sort_values(ascending=False)
            .rename("TotalMonetary")
        )
        st.markdown("**Revenue Contribution by Segment**")
        st.bar_chart(segment_revenue)

    st.markdown("### Descriptive Analytics")
    desc = result_df[["Recency", "Frequency", "Monetary", "TotalItems", "AvgOrderValue"]].describe().round(2)
    st.dataframe(desc, use_container_width=True)


def render_model_summary():
    st.markdown("## Model Summary & Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Clustering Model")
        st.write("**Selected Model:** K-Means")
        st.write("- Highest Silhouette Score")
        st.write("- Lowest Davies-Bouldin Index")
        st.write("- Highest Calinski-Harabasz Score")

        st.markdown("**Evaluation Metrics:**")
        st.write("- Silhouette Score: 0.434")
        st.write("- Davies-Bouldin Index: 0.809")
        st.write("- Calinski-Harabasz Score: 3635.78")

    with col2:
        st.markdown("### Prediction Model")
        st.write("**Model:** Random Forest Classifier")
        st.markdown("**Performance:**")
        st.write("- Accuracy: 0.698")
        st.write("- ROC-AUC: 0.718")
        st.markdown("**Interpretation:**")
        st.write("The model provides reliable predictions for customer retention and marketing strategies.")


def render_business_insights(result_df):
    st.markdown("### Business Insights")
    segment_summary = (
        result_df.groupby("SegmentName")
        .agg(
            Customers=("SegmentName", "size"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            Revenue=("Monetary", "sum"),
        )
        .round(2)
        .sort_values("Revenue", ascending=False)
    )

    st.dataframe(segment_summary, use_container_width=True)

    if "High-Value Customers" in segment_summary.index:
        hv_share = (segment_summary.loc["High-Value Customers", "Revenue"] / segment_summary["Revenue"].sum()) * 100
        st.info(
            f"High-Value Customers contribute approximately {hv_share:.1f}% of total monetary value in the uploaded data. "
            "This segment should be prioritised for retention and premium marketing initiatives."
        )

    if "Inactive Customers" in segment_summary.index:
        inactive_count = int(segment_summary.loc["Inactive Customers", "Customers"])
        st.warning(
            f"Inactive Customers account for {inactive_count:,} customers in this dataset. "
            "A win-back campaign may help recover lost engagement."
        )


def downloadable_csv(df):
    return df.to_csv(index=False).encode("utf-8")


st.markdown(
    """
    <div class="hero-card">
        <h1>🛒 E-commerce Customer Intelligence Dashboard</h1>
        <p>
            Segment customers from purchasing patterns, predict repeat purchase likelihood,
            and generate business-ready insights for marketing and retention strategy.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuration")
    st.write("This application loads trained artefacts from the `saved_models` folder.")
    st.code(
        "saved_models/\n"
        "├─ kmeans_model.pkl\n"
        "├─ scaler.pkl\n"
        "├─ future_purchase_model.pkl\n"
        "├─ selected_features.pkl\n"
        "└─ segment_names.pkl",
        language="text",
    )
    st.markdown(
        '<p class="small-note">Run the notebook first so these files are available.</p>',
        unsafe_allow_html=True,
    )

try:
    kmeans, scaler, future_model, selected_features, segment_names = load_artifacts()
except Exception as e:
    st.error("Model files could not be loaded. Please verify the `saved_models` folder and file names.")
    st.exception(e)
    st.stop()

dashboard_tab, single_tab, batch_tab, model_tab = st.tabs(
    ["Executive Dashboard", "Single Customer Prediction", "Batch Prediction", "Model Summary"]
)

with dashboard_tab:
    st.markdown("## Executive Dashboard")
    st.write(
        "Upload a CSV file with: CustomerID, Recency, Frequency, Monetary, TotalItems, AvgOrderValue "
        "(optional: FuturePurchase)."
    )

    dashboard_file = st.file_uploader(
        "Upload customer feature CSV",
        type=["csv"],
        key="dashboard_upload",
        help="Required columns: CustomerID, Recency, Frequency, Monetary, TotalItems, AvgOrderValue. Optional: FuturePurchase",
    )

    if dashboard_file is not None:
        df_dash = pd.read_csv(dashboard_file)
        is_valid, missing_cols = validate_uploaded_dataframe(df_dash)

        if not is_valid:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            result_dash, uploaded_accuracy = run_batch_predictions(
                df_dash, scaler, kmeans, future_model, selected_features, segment_names
            )

            render_kpi_row(result_dash)

            if uploaded_accuracy is not None:
                st.metric("Prediction Accuracy on Uploaded FuturePurchase Values", f"{uploaded_accuracy:.2%}")

            render_segment_analytics(result_dash)
            render_business_insights(result_dash)

            st.markdown("### Full Analysed Dataset")
            st.dataframe(result_dash, use_container_width=True)

            st.download_button(
                "Download Dashboard Results CSV",
                data=downloadable_csv(result_dash),
                file_name="dashboard_customer_predictions.csv",
                mime="text/csv",
            )
    else:
        st.markdown(
            """
            <div class="section-card">
                <h4>What this dashboard shows</h4>
                <ul>
                    <li>Customer segment distribution</li>
                    <li>Future purchase outlook</li>
                    <li>Revenue contribution by segment</li>
                    <li>Descriptive analytics for key behavioural variables</li>
                    <li>Business-oriented retention and marketing insights</li>
                    <li>Customer-level traceability using CustomerID</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

with single_tab:
    st.markdown("## Single Customer Prediction")
    st.write("Enter customer behaviour values below to predict the segment and future purchase likelihood.")

    customer_id = st.text_input("CustomerID", value="CUST_001", help="Customer identifier for display and output reference")

    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        recency = st.number_input("Recency", min_value=0.0, value=30.0, step=1.0, help="Days since last purchase")
    with input_col2:
        frequency = st.number_input("Frequency", min_value=0.0, value=5.0, step=1.0, help="Number of purchases")
    with input_col3:
        monetary = st.number_input("Monetary", min_value=0.0, value=500.0, step=10.0, help="Total amount spent")

    input_col4, input_col5 = st.columns(2)
    with input_col4:
        total_items = st.number_input("TotalItems", min_value=0.0, value=20.0, step=1.0, help="Total items purchased")
    with input_col5:
        avg_order_value = st.number_input("AvgOrderValue", min_value=0.0, value=100.0, step=1.0, help="Average order value")

    if st.button("Run Single Prediction", type="primary"):
        segment_input = prepare_segment_input(recency, frequency, monetary)
        scaled_segment_input = scaler.transform(segment_input)
        cluster = int(kmeans.predict(scaled_segment_input)[0])
        segment_name = segment_names.get(cluster, f"Segment {cluster}")

        future_input = prepare_future_input(
            recency, frequency, monetary, total_items, avg_order_value, selected_features
        )
        future_pred = int(future_model.predict(future_input)[0])
        future_label = predict_future_label(future_pred)

        left, right = st.columns([1.15, 1])

        with left:
            st.markdown("### Prediction Outcome")
            m1, m2 = st.columns(2)
            m1.metric("Predicted Segment", segment_name)
            m2.metric("Cluster ID", cluster)

            strategy = get_segment_strategy(segment_name)
            st.markdown("### Segment Interpretation")
            st.write(strategy["summary"])
            st.markdown("### Recommended Actions")
            for action in strategy["actions"]:
                st.write(f"- {action}")

        with right:
            st.markdown("### Future Purchase Outlook")
            st.metric("Prediction", future_label)

            if hasattr(future_model, "predict_proba"):
                probability = float(future_model.predict_proba(future_input)[0][1])
                st.metric("Probability of Future Purchase", f"{probability:.2%}")
                prob_df = pd.DataFrame({"Value": [probability]}, index=["Future Purchase Probability"])
                st.progress(min(max(probability, 0.0), 1.0))
                st.bar_chart(prob_df)

        st.markdown("### Customer Input Summary")
        single_result = pd.DataFrame(
            {
                "CustomerID": [customer_id],
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary],
                "TotalItems": [total_items],
                "AvgOrderValue": [avg_order_value],
                "Cluster": [cluster],
                "SegmentName": [segment_name],
                "FuturePurchasePrediction": [future_label],
            }
        )
        st.dataframe(single_result, use_container_width=True)

with batch_tab:
    st.markdown("## Batch Prediction")
    st.write(
        "Upload a CSV file with the following columns: "
        "`CustomerID`, `Recency`, `Frequency`, `Monetary`, `TotalItems`, and `AvgOrderValue` "
        "(`FuturePurchase` is optional)."
    )

    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        is_valid, missing_cols = validate_uploaded_dataframe(df)
        if not is_valid:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            result_df, uploaded_accuracy = run_batch_predictions(
                df, scaler, kmeans, future_model, selected_features, segment_names
            )

            render_kpi_row(result_df)

            if uploaded_accuracy is not None:
                st.metric("Prediction Accuracy on Uploaded FuturePurchase Values", f"{uploaded_accuracy:.2%}")

            preview_left, preview_right = st.columns(2)
            with preview_left:
                st.markdown("### Segment Mix")
                st.bar_chart(result_df["SegmentName"].value_counts())

            with preview_right:
                st.markdown("### Future Purchase Mix")
                st.bar_chart(result_df["FuturePurchasePrediction"].value_counts())

            st.markdown("### Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "Download Prediction Results CSV",
                data=downloadable_csv(result_df),
                file_name="customer_predictions.csv",
                mime="text/csv",
            )

with model_tab:
    render_model_summary()

st.markdown("---")
st.caption("Developed for E-commerce Customer Segmentation and Prediction using Streamlit.")
