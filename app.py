import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="centered"
)

# =============================================
# BACKGROUND IMAGE AND STYLING
# =============================================
def add_bg_image():
    try:
        with open("img.jpeg", "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        bg_style = f"url('data:image/jpeg;base64,{encoded}')"
    except FileNotFoundError:
        bg_style = "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

        .stApp {{
            background: {bg_style};
            background-size: cover;
            background-attachment: fixed;
        }}

        .main .block-container {{
            background-color: rgba(10, 14, 26, 0.88);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(99, 179, 237, 0.15);
            margin-top: 2rem;
            margin-bottom: 2rem;
        }}

        h1, h2, h3 {{
            font-family: 'Space Mono', monospace !important;
            color: #63b3ed !important;
            letter-spacing: -0.5px;
        }}

        p, label, .stMarkdown, div {{
            font-family: 'DM Sans', sans-serif !important;
            color: #e2e8f0 !important;
        }}

        .stButton>button {{
            background: linear-gradient(135deg, #3182ce, #2b6cb0);
            color: white !important;
            border-radius: 8px;
            padding: 0.5rem 1.4rem;
            border: none;
            font-family: 'Space Mono', monospace;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
            transition: all 0.25s ease;
        }}

        .stButton>button:hover {{
            background: linear-gradient(135deg, #4299e1, #3182ce);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 179, 237, 0.3);
        }}

        .stTextInput>div>div>input {{
            background-color: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(99, 179, 237, 0.3) !important;
            border-radius: 8px !important;
            color: #e2e8f0 !important;
        }}

        .stMetric {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid rgba(99,179,237,0.15);
        }}

        .stDataFrame {{
            border-radius: 10px;
            overflow: hidden;
        }}

        .stSelectbox>div>div {{
            background-color: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(99, 179, 237, 0.3) !important;
            color: #e2e8f0 !important;
        }}

        .stAlert {{
            border-radius: 8px;
        }}

        .stExpander {{
            border: 1px solid rgba(99,179,237,0.2) !important;
            border-radius: 10px !important;
            background: rgba(255,255,255,0.03) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

# =============================================
# SUPABASE DATABASE FUNCTIONS
# =============================================
def get_supabase_client():
    """Get Supabase client using secrets."""
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Could not connect to database. Check your Supabase secrets: {e}")
        return None


def register_user(username, password):
    supabase = get_supabase_client()
    if not supabase:
        return False
    try:
        # Check if user already exists
        result = supabase.table("users").select("username").eq("username", username).execute()
        if result.data:
            st.warning("⚠ Username already exists! Please choose another.")
            return False
        supabase.table("users").insert({"username": username, "password": password}).execute()
        return True
    except Exception as e:
        st.error(f"Registration error: {e}")
        return False


def login_user(username, password):
    supabase = get_supabase_client()
    if not supabase:
        return None
    try:
        result = supabase.table("users").select("*").eq("username", username).eq("password", password).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        st.error(f"Login error: {e}")
        return None


def get_predictions(username):
    supabase = get_supabase_client()
    if not supabase:
        return []
    try:
        result = (
            supabase.table("predictions")
            .select("stock_symbol, predicted_price, prediction_date")
            .eq("username", username)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return [(r["stock_symbol"], r["predicted_price"], r["prediction_date"]) for r in result.data]
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return []


def save_prediction(username, stock_symbol, predicted_price, prediction_date):
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        supabase.table("predictions").insert({
            "username": username,
            "stock_symbol": stock_symbol,
            "predicted_price": float(predicted_price),
            "prediction_date": prediction_date
        }).execute()
    except Exception as e:
        st.error(f"Error saving prediction: {e}")


# =============================================
# LIGHTWEIGHT ML PREDICTION
# =============================================
def predict_with_sklearn(data_scaled, days_to_predict, scaler, lookback=60):
    """
    Replaces TensorFlow LSTM with scikit-learn MLPRegressor.
    Much lighter, fast to train, no GPU required.
    """
    X, y = [], []
    for i in range(lookback, len(data_scaled) - days_to_predict):
        X.append(data_scaled[i - lookback:i, 0])
        y.append(data_scaled[i:i + days_to_predict, 0])

    X, y = np.array(X), np.array(y)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X, y)

    last_lookback = data_scaled[-lookback:, 0].reshape(1, -1)
    pred_scaled = model.predict(last_lookback)[0]
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return predictions[:days_to_predict]


# =============================================
# SESSION STATE INIT
# =============================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.selected_stock = ""
    st.session_state.prediction_data = None

# =============================================
# MAIN APP
# =============================================
st.title("📈 Stock Price Predictor")

if not st.session_state.logged_in:
    option = st.radio("Select an option", ["Sign In", "Create Account"], horizontal=True)

    if option == "Create Account":
        with st.form("create_account"):
            new_user = st.text_input("Enter Username")
            new_pass = st.text_input("Enter Password", type="password")
            if st.form_submit_button("Create Account"):
                if new_user and new_pass:
                    if register_user(new_user, new_pass):
                        st.success("✅ Account Created! Please Sign In.")
                else:
                    st.warning("Please enter both username and password.")

    elif option == "Sign In":
        with st.form("sign_in"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In"):
                user = login_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"👋 Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Credentials")

else:
    import yfinance as yf
    from yahooquery import search

    st.subheader(f"👋 Welcome back, {st.session_state.username}!")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.selected_stock = ""
        st.session_state.prediction_data = None
        st.success("Logged out successfully!")
        st.rerun()

    with st.expander("🔍 Search for Stocks", expanded=True):
        company_name = st.text_input("Enter Company Name", key="search_input")
        if st.button("Search", key="search_btn"):
            with st.spinner("🔍 Searching for companies..."):
                results = search(company_name)
                if "quotes" in results and results['quotes']:
                    stock_options = {
                        res['shortname']: res['symbol']
                        for res in results['quotes']
                        if 'symbol' in res and 'shortname' in res
                    }
                    if stock_options:
                        selected_company = st.selectbox("Select Company", list(stock_options.keys()))
                        if selected_company:
                            st.session_state.selected_stock = stock_options[selected_company]
                            st.success(f"✅ Selected: {selected_company} ({stock_options[selected_company]})")
                    else:
                        st.error("No valid companies found. Try another name.")
                else:
                    st.error("No results found. Try another name.")

    if st.session_state.selected_stock:
        st.markdown(f"### 📊 Analyzing: **{st.session_state.selected_stock}**")
        days_to_predict = st.slider("Select number of days to predict", min_value=1, max_value=30, value=7)

        if st.button("🚀 Predict Future Prices", key="predict_btn", type="primary"):
            stock_symbol = st.session_state.selected_stock

            with st.spinner("📥 Downloading historical stock data..."):
                try:
                    stock_data = yf.download(stock_symbol, period="6mo", progress=False)
                    if stock_data.empty:
                        st.error("⚠ No data found. Please check the stock symbol.")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Download failed: {str(e)}")
                    st.stop()

            st.success("✅ Historical data retrieved!")

            with st.spinner("⚙️ Processing and training model..."):
                data = stock_data[['Close']].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(data)

                predictions = predict_with_sklearn(data_scaled, days_to_predict, scaler)

                prediction_df = pd.DataFrame(predictions, columns=["Predicted Price"])
                prediction_df.index = pd.date_range(
                    start=pd.to_datetime(stock_data.index[-1]) + pd.Timedelta(days=1),
                    periods=len(predictions),
                    freq='B'
                )

                st.session_state.prediction_data = {
                    'df': prediction_df,
                    'last_price': float(data[-1][0]),
                    'predicted_price': float(predictions[-1][0]),
                    'stock_symbol': stock_symbol,
                    'prediction_date': str(prediction_df.index[-1].date())
                }

            save_prediction(
                st.session_state.username,
                stock_symbol,
                float(predictions[-1][0]),
                str(prediction_df.index[-1].date())
            )

            st.success("🎉 Prediction complete!")
            st.rerun()

        if st.session_state.prediction_data:
            st.line_chart(st.session_state.prediction_data['df'])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Last Actual Price", f"₹{st.session_state.prediction_data['last_price']:.2f}")
            with col2:
                st.metric(
                    f"Predicted Price (Day {days_to_predict})",
                    f"₹{st.session_state.prediction_data['predicted_price']:.2f}"
                )

    if st.checkbox("📜 Show My Prediction History"):
        history = get_predictions(st.session_state.username)
        if history:
            df = pd.DataFrame(history, columns=["Stock", "Predicted Price", "Date"])
            st.dataframe(
                df.style.format({"Predicted Price": "₹{:.2f}"}),
                height=400
            )
        else:
            st.info("No prediction history found.")
