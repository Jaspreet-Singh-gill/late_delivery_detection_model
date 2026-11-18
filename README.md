# Late Delivery Risk Prediction in Supply Chain

A machine learning system predicting late delivery risk at the time of order placement using the DataCo Supply Chain dataset. Built for Data Science course evaluation, presentation, and viva defense.

---

## 1. Project Overview & Motivation

In modern e-commerce and supply chain logistics, late deliveries lead directly to customer dissatisfaction, negative reviews, customer support surges, and permanent churn. Most logistics tracking systems only detect delays while a package is already in transit—when corrective options are limited and expensive.

### Objective
Predict whether an incoming order will be delivered late (`Late_delivery_risk = 1`) **at the exact moment of order placement/dispatch commitment**, relying strictly on information known at that time.

### Core Principle: Zero Target Leakage
Many existing academic or Kaggle projects on DataCo report artificial accuracy (>98%) by leaking post-delivery columns (`Days for shipping (real)` or `Delivery Status`). In this project, all post-order and outcome features are audited, flagged, and strictly dropped. The resulting models achieve realistic, defensible performance suitable for a technical viva defense.

---

## 2. Dataset Information

- **Dataset**: DataCo Supply Chain Dataset (`DataCoSupplyChainDataset.csv`)
- **Encoding**: `latin-1`
- **Total Records**: `180,519` rows
- **Raw Features**: `53` columns
- **Date Range**: January 1, 2015 to January 31, 2018 (3 full years)
- **Target Variable**: `Late_delivery_risk`
  - `1` (Late): `98,977` orders (**54.83%**)
  - `0` (On-time / Early): `81,542` orders (**45.17%**)
  - Near-balanced distribution; no artificial oversampling (SMOTE) required.

---

## 3. Exploratory Data Analysis (EDA) Findings

Detailed EDA was conducted in [`notebooks/01_eda.ipynb`](file:///notebooks/01_eda.ipynb). All figures below were computed directly on the dataset:

### A. Missing Values & Data Hygiene
- `Product Description`: 100% missing (`180,519` nulls) - dropped.
- `Order Zipcode`: 86.24% missing (`155,679` nulls) - dropped.
- `Customer Lname`: 8 missing (<0.01%).
- `Customer Zipcode`: 3 missing (<0.01%).
- `Duplicates`: Exactly `0` duplicate rows across all 180,519 records.

### B. Severe Target Leakage Uncovered
Four feature groups were identified as post-delivery leakage and removed:
1. **`Delivery Status`**:
   - `Late delivery`: 98,977 rows — 100% have `Late_delivery_risk = 1`.
   - `Advance shipping` (41,592), `Shipping on time` (32,196), `Shipping canceled` (7,754) — 100% have `Late_delivery_risk = 0`.
   - *Finding*: `Delivery Status` is a direct text representation of the target.
2. **`Days for shipping (real)`**:
   - The actual transit duration recorded only after arrival. Pearson correlation with target = `+0.4014`.
3. **`shipping date (DateOrders)`**:
   - Equal to `order date + Days for shipping (real)`.
4. **`Order Status`**:
   - Categories like `SUSPECTED_FRAUD` (2,316) and `CANCELED` (2,107) are post-placement operational actions. Because canceled shipments are marked risk = 0, keeping order status leaks cancellation outcomes.

### C. Redundant & Identical Columns Dropped
Six pairs of columns are 100% redundant duplicates:
- `Benefit per order` == `Order Profit Per Order`
- `Sales per customer` == `Order Item Total`
- `Product Price` == `Order Item Product Price`
- `Customer Id` == `Order Customer Id`
- `Product Card Id` == `Order Item Cardprod Id`
- `Category Id` == `Product Category Id`

### D. Surprising Finding: Shipping Mode Dominance
`Shipping Mode` has a deterministic 1:1 mapping with `Days for shipment (scheduled)`:
- `Same Day`: scheduled = `0` days
- `First Class`: scheduled = `1` day
- `Second Class`: scheduled = `2` days
- `Standard Class`: scheduled = `4` days

**Late Rate by Shipping Mode**:
- `First Class` (27,814 orders): **95.32% late**
- `Second Class` (35,216 orders): **76.63% late**
- `Same Day` (9,737 orders): **45.74% late**
- `Standard Class` (107,752 orders): **38.07% late**

**Root Operational Cause**:
Cross-tabulating `Days for shipping (real)` vs `Shipping Mode` revealed why:
- **First Class**: Promises delivery in 1 day. In reality, *every single First Class shipment takes 2 days* (100% of non-canceled orders). Because 2 > 1, 95.3% are late.
- **Second Class**: Promises 2 days. Transit takes 2, 3, 4, 5, or 6 days evenly distributed (~20% each). Deliveries taking 3 to 6 days are late -> 76.6% late rate.
- **Same Day**: Promises 0 days. Real transit takes 0 days (52.2%) or 1 day (47.8%) -> 45.7% late rate.
- **Standard Class**: Promises 4 days. Real transit takes 2, 3, 4, 5, or 6 days evenly. Days 2, 3, and 4 are on-time (60%), while days 5 and 6 miss the SLA (40%) -> 38.1% late rate.

### E. Distribution Across Other Dimensions
- **Market**: Africa (`54.59%`), Europe (`55.21%`), LATAM (`54.36%`), Pacific Asia (`55.05%`), USCA (`54.80%`).
- **Customer Segment**: Consumer (`54.81%`), Corporate (`54.72%`), Home Office (`55.07%`).
- **Temporal Stability**:
  - Month: `54.06%` (July) to `55.79%` (August).
  - Day of Week: `54.30%` (Tuesday) to `55.17%` (Friday).
  - Hour of Day: `54.2%` to `55.6%` across all 24 hours.
- **Numeric Features**:
  - `Sales`, `Quantity`, `Discount Rate`, and `Price` have Pearson correlations between `-0.014` and `+0.003`.

---

## 4. Feature Decisions & Time-Based Split

Implemented in [`notebooks/02_features.ipynb`](file:///notebooks/02_features.ipynb):

### Selected Features (19 Order-Time Predictors)
- **Categorical (7)**: `Shipping Mode`, `Customer Segment`, `Market`, `Order Region`, `Department Name`, `Category Name`, `Type` (payment method).
- **Numerical & Engineered (12)**:
  - `Order Item Quantity`, `Order Item Product Price`, `Order Item Discount Rate`, `Order Item Discount`, `Order Item Total`, `Sales`.
  - `order_month` (1-12): Seasonal carrier capacity strains.
  - `order_day` (1-31): Month-end fulfillment surges.
  - `order_dow` (0-6): Day-of-week dispatch schedules.
  - `order_hour` (0-23): Pre/post daily carrier pickup cutoffs.
  - `is_weekend` (0 or 1): Orders placed on Saturday/Sunday.
  - `is_domestic` (0 or 1): Domestic US/PR delivery vs international export.

### Time-Based Split Justification
To simulate real-world deployment, the dataset was sorted chronologically by order timestamp:
- **Training Set (Earliest 70%)**: `126,363` orders (Jan 1, 2015 – Jan 8, 2017) | Late Rate: `54.94%`
- **Validation Set (Next 15%)**: `27,078` orders (Jan 8, 2017 – Jun 14, 2017) | Late Rate: `54.11%`
- **Final Holdout Set (Final 15%)**: `27,078` orders (Jun 14, 2017 – Jan 31, 2018) | Late Rate: `55.02%`

*Why not a random split?* A random split leaks seasonal trends, carrier disruptions, and same-week orders into the test set, creating optimistic data snooping. Time-based splitting evaluates model generalization to future quarters.

### Preprocessing Pipeline (`ColumnTransformer`)
- Categorical features: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- Numerical features: `StandardScaler()`
- **Zero Leakage**: Fit strictly on `X_train`, saved to `models/preprocessor.joblib`.

---

## 5. Model Results & Comparison

Implemented in [`notebooks/03_models.ipynb`](file:///notebooks/03_models.ipynb). Evaluated on the validation set (`27,078` orders) at default threshold `0.50`:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.6947 | 0.8393 | 0.5390 | 0.6565 | 0.7292 | 0.8035 |
| **Decision Tree** | 0.7080 | 0.8227 | 0.5869 | 0.6851 | 0.7566 | 0.8161 |
| **Random Forest** | 0.7117 | 0.8430 | 0.5741 | 0.6830 | 0.7568 | 0.8206 |
| **XGBoost (Best)** | **0.7101** | **0.8375** | **0.5760** | **0.6825** | **0.7585** | **0.8237** |

### Decision Threshold Tuning (Why 0.40?)
In supply chain operations, the cost of a **False Negative (missed delay)** is high (customer churn, SLA fines), while the cost of a **False Positive (false alarm)** is low (warehouse priority picking).

At `threshold = 0.50`, XGBoost has high precision (`83.75%`) but misses `42.4%` of late orders (recall = `57.60%`).
Shifting to `threshold = 0.40`:
- **Recall jumps from 57.60% to 69.88%** (catching over 1,800 additional late orders).
- **Precision remains high at 70.32%**.
- **F1-Score increases from 0.6825 to 0.7010**.

---

## 6. Final Holdout Evaluation (Evaluated ONCE)

The final selected XGBoost model was evaluated once on the untouched out-of-time holdout set (`data/holdout/holdout.csv`, 27,078 orders from June 2017 to January 2018):

| Metric | Threshold = 0.50 | Threshold = 0.40 (Operational) |
| :--- | :---: | :---: |
| **Accuracy** | 71.90% | 66.60% |
| **Precision** | 84.63% | 67.01% |
| **Recall** | 59.78% | **77.43%** |
| **F1-Score** | 0.7007 | **0.7184** |
| **ROC-AUC** | **0.7703** | **0.7703** |
| **PR-AUC** | **0.8374** | **0.8374** |

### Confusion Matrix on Holdout (Threshold = 0.40)
- **True Positives (Late orders correctly caught)**: `11,535` orders
- **False Negatives (Missed late orders)**: `3,363` orders
- **True Negatives (On-time orders confirmed)**: `6,500` orders
- **False Positives (False alarms)**: `5,680` orders

Notice that ROC-AUC on holdout (`0.7703`) matches the validation ROC-AUC (`0.7585`), confirming zero overfitting or temporal degradation.

---

## 7. Streamlit Web Dashboard

An interactive multi-page dashboard is implemented in [`app/streamlit_app.py`](file:///app/streamlit_app.py) with 4 pages:
1. **Overview & EDA**: High-level metrics, interactive tabs with EDA charts, leakage explanation.
2. **Model Comparison**: Metric comparison table, ROC/PR curves, confusion matrices, threshold trade-off curve, and global feature importance.
3. **Holdout Predictions**: Loads `data/holdout/holdout.csv`, dynamically computes out-of-time predictions, lets users adjust threshold, and provides a filterable table by shipping mode and market.
4. **Single Order Predictor**: An interactive form to enter order parameters, compute late probability, and view a SHAP feature contribution bar chart explaining why the order was flagged.

---

## 8. How to Run

### Setup Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Install dependencies:
pip install -r requirements.txt
```

### Reproduce Notebooks
```bash
# Execute Phase 1 (EDA):
python run_nb.py

# Execute Phase 2 (Features & Splits):
python run_nb2.py

# Execute Phase 3 (Model Training & Evaluation):
python run_nb3.py
```

### Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 9. Limitations & Practical Constraints

1. **Synthetic/Public Dataset Characteristics**: DataCo exhibits strong artifacts—specifically, First Class shipping takes 2 days 100% of the time, and Second Class transit is uniformly distributed between 2 and 6 days.
2. **Single Company Scope**: Data represents a single distributor shipping athletic/outdoor goods from Puerto Rico and the United States; results cannot be blindly transferred to international retail giants without retraining.
3. **Absence of Real-Time Weather/Traffic Data**: Predictions rely on static order-time attributes. Real-world delivery delay systems benefit from live weather, carrier hub congestion, and package GPS telematics.
