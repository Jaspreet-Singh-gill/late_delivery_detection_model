import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Resolve project base directory dynamically so the app runs from any working directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_path(rel_path):
    if os.path.exists(rel_path):
        return os.path.abspath(rel_path)
    base_rel = os.path.join(BASE_DIR, rel_path)
    if os.path.exists(base_rel):
        return base_rel
    return base_rel

st.set_page_config(
    page_title="Late Delivery Prediction Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom light styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_artifacts():
    pipeline = joblib.load(get_path("models/pipeline.joblib"))
    model = joblib.load(get_path("models/final_model.joblib"))
    preprocessor = joblib.load(get_path("models/preprocessor.joblib"))
    metadata = joblib.load(get_path("models/model_metadata.joblib"))
    return pipeline, model, preprocessor, metadata

@st.cache_data
def load_holdout_sample():
    df = pd.read_csv(get_path("data/holdout/holdout.csv"))
    return df

@st.cache_data
def load_metrics():
    val_m = pd.read_csv(get_path("reports/metrics.csv"))
    hold_m = pd.read_csv(get_path("reports/holdout_metrics.csv"))
    return val_m, hold_m

try:
    pipeline, model, preprocessor, metadata = load_models_and_artifacts()
    val_metrics, holdout_metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading models or reports: {e}. Base directory: {BASE_DIR}")
    st.stop()

# Sidebar Navigation
st.sidebar.title("📦 Logistics Intelligence")
st.sidebar.caption("DataCo Supply Chain - Late Delivery Prediction")
page = st.sidebar.radio(
    "Go to",
    ["1. Project Overview & EDA", "2. Model Comparison", "3. Final Holdout Predictions", "4. Single Order Predictor"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
- **Target**: `Late_delivery_risk`
- **Model**: Tuned XGBoost Classifier
- **Choosen**: Threshold : 0.40
""")

# ================= PAGE 1 =================
if page == "1. Project Overview & EDA":
    st.title("📦 Late Delivery Prediction: Overview & EDA")
    st.markdown("""
    Predicting delivery delays at **order creation time** enables logistics teams to intervene before failure occurs—by upgrading carrier tiers, picking orders earlier, or notifying customers proactively.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", "180,519", "Jan 2015 - Jan 2018")
    with col2:
        st.metric("Late Delivery Rate", "54.83%", "98,977 late orders")
    with col3:
        st.metric("Order-Time Features", "19", "26 leakage cols dropped")
    with col4:
        st.metric("Holdout Set Size", "27,078", "Out-of-time evaluation")

    st.markdown("---")
    st.subheader("Key Exploratory Insights")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Shipping Mode Dominance",
        "Target & Data Leakage",
        "Geographic & Temporal Rates",
        "Numeric Distributions"
    ])

    with tab1:
        st.markdown("""
        ### Why Shipping Mode Drives Lateness
        Expedited delivery modes set systematically unrealistic delivery promises compared to physical carrier transit capabilities:
        - **First Class**: Promises delivery in **1 day**, but actual transit in the data **always takes 2 days** -> **95.3% Late Rate**.
        - **Second Class**: Promises **2 days**, but transit takes 2-6 days uniformly -> **76.6% Late Rate**.
        - **Same Day**: Promises **0 days**, transit takes 0-1 days -> **45.7% Late Rate**.
        - **Standard Class**: Promises **4 days**, transit takes 2-6 days -> **38.1% Late Rate**.
        """)
        c1, c2 = st.columns(2)
        with c1:
            p = get_path("reports/late_rate_by_shipping_mode.png")
            if os.path.exists(p):
                st.image(p, caption="Late Delivery Rate by Shipping Mode")
        with c2:
            p = get_path("reports/shipping_mode_vs_real_days.png")
            if os.path.exists(p):
                st.image(p, caption="Actual Transit Days Taken per Shipping Mode")

    with tab2:
        st.markdown("""
        ### Target Leakage Audit
        Features that directly encode post-delivery outcomes were identified and eliminated:
        - `Delivery Status`: Categorical label where `'Late delivery'` is 100% synonymous with the target.
        - `Days for shipping (real)`: The actual duration recorded after the customer receives the box.
        - `shipping date (DateOrders)`: Computed post-hoc as `order date + real shipping days`.
        - `Order Status`: Post-fulfillment statuses like `SUSPECTED_FRAUD` and `CANCELED` never have a late risk flag.
        """)
        p = get_path("reports/target_distribution.png")
        if os.path.exists(p):
            st.image(p, caption="Target Class Distribution (54.8% Late vs 45.2% On-time)", width=500)

    with tab3:
        st.markdown("""
        ### Geographic and Temporal Flatness
        Unlike Shipping Mode, geographic markets (Africa, Europe, LATAM, Pacific Asia, USCA) and customer segments all hover between **54.3% and 55.2%** late rate.
        Similarly, order month (54.1% - 55.8%) and day of week show narrow variance, indicating that delays are structural logistics SLA bottlenecks rather than seasonal or regional issues.
        """)
        c1, c2 = st.columns(2)
        with c1:
            p = get_path("reports/late_rate_by_market_and_segment.png")
            if os.path.exists(p):
                st.image(p, caption="Late Rates across Markets and Segments")
        with c2:
            p = get_path("reports/temporal_patterns.png")
            if os.path.exists(p):
                st.image(p, caption="Temporal Late Rates (Month, Weekday, Hour)")

    with tab4:
        st.markdown("""
        ### Numeric Feature Correlations
        Numeric features available at order time (`Sales`, `Order Item Quantity`, `Order Item Discount Rate`, `Product Price`) have Pearson correlations between **-0.014 and +0.003** with the target.
        """)
        c1, c2 = st.columns(2)
        with c1:
            p = get_path("reports/correlation_heatmap.png")
            if os.path.exists(p):
                st.image(p, caption="Pearson Correlations with Late_delivery_risk")
        with c2:
            p = get_path("reports/numeric_boxplots.png")
            if os.path.exists(p):
                st.image(p, caption="Distributions of Numeric Features by Late Risk")

# ================= PAGE 2 =================
elif page == "2. Model Comparison":
    st.title("📊 Model Comparison & Threshold Tuning")
    st.markdown("""
    Four classification models were trained on the earliest 70% time-sorted records (`126,363` rows) and validated on the subsequent 15% out-of-time partition (`27,078` rows).
    """)

    st.subheader("Validation Set Metrics (Threshold = 0.50)")
    st.dataframe(val_metrics.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'], color='#d4edda'), width='stretch')

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        p = get_path("reports/roc_pr_curves.png")
        if os.path.exists(p):
            st.image(p, caption="ROC & Precision-Recall Curves (Validation Set)")
    with c2:
        p = get_path("reports/confusion_matrix.png")
        if os.path.exists(p):
            st.image(p, caption="Confusion Matrices at Default Threshold 0.50")

    st.markdown("---")
    st.subheader("Decision Threshold Optimization: Why 0.40?")
    st.markdown("""
    In logistics, **missing a late order (False Negative)** costs significantly more than a **false alarm (False Positive)**:
    - Missed delay -> Angry customer, customer support tickets, SLA penalty fees, churn.
    - False alarm -> Proactive warehouse priority picking or early dispatch check.
    
    By shifting threshold from **0.50 down to 0.40**:
    - **Recall surges from 57.6% to 69.9%** (catching an additional 1,800+ late orders).
    - **Precision stays solid at 70.3%**.
    - **F1-Score improves from 0.6825 to 0.7010**.
    """)

    c1, c2 = st.columns(2)
    with c1:
        p = get_path("reports/threshold_tuning.png")
        if os.path.exists(p):
            st.image(p, caption="Precision, Recall & F1 Trade-off vs. Threshold")
    with c2:
        p = get_path("reports/feature_importance.png")
        if os.path.exists(p):
            st.image(p, caption="Top 15 Feature Importances (XGBoost Gain)")

    p = get_path("reports/shap_summary.png")
    if os.path.exists(p):
        st.image(p, caption="SHAP Summary Plot (Top Predictive Drivers)")

# ================= PAGE 3 =================
elif page == "3. Final Holdout Predictions":
    st.title("🎯 Final Holdout Set Evaluation")
    st.markdown("""
    Evaluation on the **final 15% chronologically out-of-time holdout set** (`data/holdout/holdout.csv`, 27,078 rows from June 2017 to January 2018).
    This dataset was **never touched** during training or hyperparameter tuning.
    """)

    holdout_df = load_holdout_sample()
    feature_cols = metadata['all_feature_cols']
    target = 'Late_delivery_risk'

    # Compute predictions
    with st.spinner("Generating holdout predictions using cached pipeline..."):
        probs = pipeline.predict_proba(holdout_df[feature_cols])[:, 1]

    # Sidebar / interactive threshold
    selected_th = st.slider("Select Decision Threshold", min_value=0.20, max_value=0.70, value=0.40, step=0.05)
    preds = (probs >= selected_th).astype(int)
    y_true = holdout_df[target].values

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    roc_auc = roc_auc_score(y_true, probs)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Holdout Accuracy", f"{acc*100:.2f}%")
    c2.metric("Holdout Precision", f"{prec*100:.2f}%")
    c3.metric("Holdout Recall", f"{rec*100:.2f}%", f"{int(np.sum((preds==1)&(y_true==1))):,} caught")
    c4.metric("Holdout F1-Score", f"{f1:.4f}")
    c5.metric("Holdout ROC-AUC", f"{roc_auc:.4f}")

    st.markdown("---")
    col_cm, col_filt = st.columns([1, 2])

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['On-Time (0)', 'Late (1)'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title(f"Holdout Confusion Matrix (Th={selected_th})")
        st.pyplot(fig)

    with col_filt:
        st.subheader("Filterable Holdout Predictions")
        holdout_display = holdout_df.copy()
        holdout_display['Late_Probability'] = np.round(probs, 4)
        holdout_display['Predicted_Late'] = preds
        holdout_display['Actual_Late'] = y_true

        # Filters
        f_mode = st.selectbox("Filter Shipping Mode", ["All"] + list(holdout_df['Shipping Mode'].unique()))
        f_market = st.selectbox("Filter Market", ["All"] + list(holdout_df['Market'].unique()))

        filtered = holdout_display
        if f_mode != "All":
            filtered = filtered[filtered['Shipping Mode'] == f_mode]
        if f_market != "All":
            filtered = filtered[filtered['Market'] == f_market]

        show_cols = ['Shipping Mode', 'Market', 'Order Region', 'Department Name', 'Order Item Quantity', 'Sales', 'Late_Probability', 'Predicted_Late', 'Actual_Late']
        st.dataframe(filtered[show_cols].head(100), width='stretch')
        st.caption(f"Showing 100 rows out of {len(filtered):,} filtered holdout orders.")

# ================= PAGE 4 =================
elif page == "4. Single Order Predictor":
    st.title("⚡ Single Order Late Delivery Predictor")
    st.markdown("Enter details for a new incoming order to calculate its late delivery risk and view feature explanations.")

    with st.form("single_order_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Shipping & Logistics")
            shipping_mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
            market = st.selectbox("Market", ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"])
            order_region = st.selectbox("Order Region", [
                "Western Europe", "Central America", "South America", "Oceania",
                "Northern Europe", "Southeast Asia", "Southern Europe", "Caribbean",
                "West of USA ", "South Asia", "Eastern Europe", "North Africa"
            ])
            is_domestic = st.checkbox("Domestic Shipment (USA/Puerto Rico destination)", value=False)

        with col2:
            st.subheader("Item & Pricing")
            department = st.selectbox("Department", ["Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors", "Fitness"])
            category = st.selectbox("Category", [
                "Cleats", "Men's Footwear", "Women's Apparel", "Indoor/Outdoor Games",
                "Fishing", "Water Sports", "Camping & Hiking", "Cardio Equipment"
            ])
            payment_type = st.selectbox("Payment Type", ["DEBIT", "TRANSFER", "PAYMENT", "CASH"])
            quantity = st.slider("Order Item Quantity", 1, 5, 1)
            product_price = st.number_input("Unit Product Price ($)", min_value=5.0, max_value=2000.0, value=59.99, step=5.0)
            discount_rate = st.slider("Discount Rate", min_value=0.0, max_value=0.25, value=0.05, step=0.01)

        with col3:
            st.subheader("Order Timing & Threshold")
            order_month = st.slider("Order Month", 1, 12, 10)
            order_day = st.slider("Order Day of Month", 1, 31, 15)
            order_dow = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            order_hour = st.slider("Order Hour (0-23)", 0, 23, 14)
            decision_th = st.slider("Operational Decision Threshold", 0.20, 0.70, 0.40, 0.05)

        submit_btn = st.form_submit_button("Predict Delivery Risk")

    if submit_btn:
        dow_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        dow_idx = dow_mapping[order_dow]
        is_weekend = 1 if dow_idx >= 5 else 0

        # Calculations
        sales = round(quantity * product_price, 2)
        discount_amount = round(sales * discount_rate, 2)
        item_total = round(sales - discount_amount, 2)

        input_data = pd.DataFrame([{
            'Shipping Mode': shipping_mode,
            'Customer Segment': 'Consumer',
            'Market': market,
            'Order Region': order_region,
            'Department Name': department,
            'Category Name': category,
            'Type': payment_type,
            'Order Item Quantity': quantity,
            'Order Item Product Price': product_price,
            'Order Item Discount Rate': discount_rate,
            'Order Item Discount': discount_amount,
            'Order Item Total': item_total,
            'Sales': sales,
            'order_month': order_month,
            'order_day': order_day,
            'order_dow': dow_idx,
            'order_hour': order_hour,
            'is_weekend': is_weekend,
            'is_domestic': int(is_domestic)
        }])

        prob_late = pipeline.predict_proba(input_data)[0, 1]
        is_late = prob_late >= decision_th

        st.markdown("---")
        st.subheader("Prediction Result")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if is_late:
                st.error(f"⚠️ **HIGH RISK OF DELAY**")
                st.metric("Late Delivery Probability", f"{prob_late*100:.1f}%", f"Above threshold ({decision_th:.2f})")
            else:
                st.success(f"✅ **ON-TIME DELIVERY EXPECTED**")
                st.metric("Late Delivery Probability", f"{prob_late*100:.1f}%", f"Below threshold ({decision_th:.2f})")

            st.progress(float(prob_late))

        with res_col2:
            st.markdown("### Operational Recommendations")
            if is_late:
                if shipping_mode == "First Class":
                    st.warning("""
                    - **Root Cause**: First Class promises 1-day delivery, but standard fulfillment takes 2 days.
                    - **Action**: Immediately flag for priority fulfillment queue, pack within 2 hours, and dispatch on earliest morning flight.
                    """)
                elif shipping_mode == "Second Class":
                    st.warning("""
                    - **Root Cause**: Second Class promises 2-day delivery, which is missed ~76.6% of the time.
                    - **Action**: Alert customer with realistic tracking milestones or upgrade to express courier partner at zero surcharge.
                    """)
                else:
                    st.warning("""
                    - **Action**: Order flagged due to high-risk regional destination or fulfillment cutoff timing. Notify regional dispatch coordinator.
                    """)
            else:
                st.info("""
                - **Status**: Standard processing path approved. Estimated transit aligns comfortably with delivery promise.
                """)

        # Fast Feature Contribution Explanation
        st.markdown("### What Factors Influenced This Prediction?")
        import shap
        X_trans = preprocessor.transform(input_data)
        explainer = shap.TreeExplainer(model)
        shap_val = explainer(X_trans)
        
        feature_names = metadata['all_feature_names']
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': shap_val.values[0]
        }).sort_values('SHAP_Value', key=abs, ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ['crimson' if v > 0 else 'forestgreen' for v in shap_df['SHAP_Value']]
        ax.barh(shap_df['Feature'][::-1], shap_df['SHAP_Value'][::-1], color=colors[::-1])
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title("Top Feature Contributions (Positive = Pushes toward Late, Negative = Pushes toward On-Time)")
        ax.set_xlabel("SHAP Value")
        st.pyplot(fig)
