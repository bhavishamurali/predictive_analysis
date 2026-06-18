import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Set page configurations
st.set_page_config(page_title="E-Commerce Revenue Forecast", layout="wide")

# ----------------------------------------------------------------
# HEADER & INTRODUCTION
# ----------------------------------------------------------------
st.title("📊 Predictive Analytics: E-Commerce Revenue Dashboard")
st.markdown("""
This web application uses a **Random Forest Regressor** machine learning model trained on historical store data 
to forecast daily sales revenue based on marketing metrics and seasonal discounts.
""")

# ----------------------------------------------------------------
# DATA GENERATION & TRAINING (Cached for Performance)
# ----------------------------------------------------------------
@st.cache_data
def load_and_train_model():
    np.random.seed(42)
    n_days = 365
    
    marketing_spend = np.random.uniform(500, 5000, n_days)
    website_traffic = (marketing_spend * 1.5) + np.random.normal(1000, 300, n_days)
    discount_percent = np.random.choice([0, 10, 20, 30], size=n_days, p=[0.4, 0.3, 0.2, 0.1])
    
    base_revenue = 2000
    daily_revenue = (
        base_revenue 
        + (marketing_spend * 2.1) 
        + (website_traffic * 0.4) 
        + (discount_percent * 120) 
        + np.random.normal(0, 800, n_days)
    )
    
    df = pd.DataFrame({
        'Marketing_Spend': marketing_spend,
        'Website_Traffic': website_traffic,
        'Discount_Percent': discount_percent,
        'Daily_Revenue': daily_revenue
    })
    
    X = df[['Marketing_Spend', 'Website_Traffic', 'Discount_Percent']]
    y = df['Daily_Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate evaluation metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, df, mae, r2, y_test, y_pred

model, df, mae, r2, y_test, y_pred = load_and_train_model()

# ----------------------------------------------------------------
# SIDEBAR CONTROL PANEL (Interactive Forecast Inputs)
# ----------------------------------------------------------------
st.sidebar.header("🔮 Forecast Engine: Input Features")
st.sidebar.markdown("Adjust the sliders below to predict daily revenue:")

input_marketing = st.sidebar.slider("Daily Marketing Spend ($)", min_value=500, max_value=5000, value=2500, step=100)
input_traffic = st.sidebar.slider("Estimated Website Traffic (Users)", min_value=1000, max_value=10000, value=4500, step=250)
input_discount = st.sidebar.selectbox("Active Store Discount (%)", options=[0, 10, 20, 30], index=1)

# Predict based on user inputs
user_features = pd.DataFrame([[input_marketing, input_traffic, input_discount]], 
                             columns=['Marketing_Spend', 'Website_Traffic', 'Discount_Percent'])
predicted_revenue = model.predict(user_features)[0]

# ----------------------------------------------------------------
# INTERACTIVE DASHBOARD LAYOUT
# ----------------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🚀 Live Prediction")
    st.metric(label="Forecasted Daily Revenue", value=f"${predicted_revenue:,.2f}")
    
    st.write("---")
    st.subheader("📉 Model Performance Metrics")
    st.metric(label="R-squared (Accuracy)", value=f"{r2:.2%}")
    st.metric(label="Mean Absolute Error", value=f"${mae:.2f}")

with col2:
    st.subheader("📈 Model Evaluation Scatter Plot")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='teal', ax=ax, s=60)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='darkorange', lw=2, linestyle='--')
    
    ax.set_xlabel("Actual Historical Revenue ($)")
    ax.set_ylabel("Predicted Model Revenue ($)")
    st.pyplot(fig)

# Show raw historical data preview at the bottom
st.write("---")
st.subheader("📋 Historical Training Data Sample")
st.dataframe(df.head(10), use_container_width=True)