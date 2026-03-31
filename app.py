# ==========================================
# 🚗 Car Price Prediction Web App (Advanced UI)
# ==========================================

import streamlit as st
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ------------------------------------------
# 🎨 Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ------------------------------------------
# 🎨 Custom CSS for Attractive UI
# ------------------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
    }
    .main {
        background: transparent;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# 🚗 Title
# ------------------------------------------
st.markdown('<p class="title">🚗 Car Price Prediction App</p>', unsafe_allow_html=True)

# ------------------------------------------
# 📂 Load Dataset
# ------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("dataset.csv")
    data = data.drop(['car_ID'], axis=1)
    return data

dataset = load_data()

# ------------------------------------------
# 🎯 Prepare Data
# ------------------------------------------
Xdata = dataset.drop('price', axis=1)
numericalCols = Xdata.select_dtypes(exclude=['object']).columns
X = Xdata[numericalCols]
Y = dataset['price']

# Scaling
cols = X.columns
X = pd.DataFrame(scale(X), columns=cols)

# Train model
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = RandomForestRegressor()
model.fit(x_train, y_train)

# ------------------------------------------
# 🧠 User Input Section
# ------------------------------------------
st.sidebar.header("🔧 Enter Car Details")

user_input = {}

for col in cols:
    user_input[col] = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

input_df = pd.DataFrame([user_input])

# ------------------------------------------
# 🔮 Prediction
# ------------------------------------------
if st.sidebar.button("Predict Price 🚀"):
    prediction = model.predict(input_df)

    st.markdown(f"""
        <div class="card">
            <h2>💰 Estimated Car Price</h2>
            <h1>₹ {prediction[0]:,.2f}</h1>
        </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# 📊 Dataset Preview
# ------------------------------------------
with st.expander("📊 View Dataset"):
    st.dataframe(dataset)