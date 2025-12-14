import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioCare AI | Heart Risk Assessment",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background: linear-gradient(to bottom right, #f8f9fa, #e9ecef); font-family: 'Segoe UI', sans-serif; }
    .main-header { font-size: 3rem; color: #c92a2a; font-weight: 800; text-align: center; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .sub-header { font-size: 1.2rem; color: #495057; text-align: center; margin-bottom: 2rem; }
    .input-card { background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; border-top: 4px solid #c92a2a; }
    .card-title { font-size: 1.2rem; font-weight: 700; color: #343a40; margin-bottom: 15px; border-bottom: 2px solid #f1f3f5; padding-bottom: 10px; }
    .stButton>button { width: 100%; background: linear-gradient(90deg, #c92a2a 0%, #e03131 100%); color: white; font-weight: 600; font-size: 1.2rem; padding: 12px; border-radius: 10px; border: none; box-shadow: 0 4px 10px rgba(201, 42, 42, 0.3); transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(201, 42, 42, 0.4); }
    .result-box-safe { background-color: #d3f9d8; color: #2b8a3e; padding: 25px; border-radius: 12px; text-align: center; border: 2px solid #b2f2bb; }
    .result-box-danger { background-color: #ffe3e3; color: #c92a2a; padding: 25px; border-radius: 12px; text-align: center; border: 2px solid #ffc9c9; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SMART MODEL LOADER (Auto-Trains if Missing)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_data():
    file_path = "heart_disease_2020_model.pkl"
    
    # If file exists, load it
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    
    # If file MISSING, train it now (Lazy Training)
    else:
        status = st.empty()
        status.info("⚙️ Model not found. Downloading dataset and training first-time setup... (This takes ~30 seconds)")
        
        try:
            import kagglehub
            
            # 1. Download Data
            path = kagglehub.dataset_download("aqleemkhan/heart-disease-2020")
            csv_path = os.path.join(path, "heart_2020_cleaned.csv")
            
            # Find csv if name differs
            if not os.path.exists(csv_path):
                for f in os.listdir(path):
                    if f.endswith(".csv"):
                        csv_path = os.path.join(path, f)
                        break
            
            dia = pd.read_csv(csv_path)
            
            # 2. Encode Data
            LE = {}
            for col in dia.columns:
                if dia[col].dtype == 'object':
                    LE[col] = LabelEncoder()
                    dia[col] = LE[col].fit_transform(dia[col])
            
            # 3. Train Model
            ind = dia.drop('HeartDisease', axis=1)
            dep = dia['HeartDisease']
            X_train, _, y_train, _ = train_test_split(ind, dep, test_size=0.2, random_state=42)
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # 4. Save and Return
            data = {"model": model, "encoders": LE}
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            
            status.success("✅ Training Complete! App is ready.")
            time.sleep(1)
            status.empty()
            return data
            
        except Exception as e:
            status.error(f"Error during training: {e}")
            return None

data = get_model_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=80)
    st.title("CardioCare AI")
    st.markdown("---")
    st.info("This AI assesses cardiovascular risk using a Logistic Regression model trained on CDC data.")
    st.write("---")
    st.caption("⚠️ **Disclaimer:** Educational tool only. Not medical advice.")

# -----------------------------------------------------------------------------
# 4. MAIN INTERFACE
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">💓 Heart Disease Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Please fill in the health parameters below for a comprehensive analysis.</div>', unsafe_allow_html=True)

if data:
    model = data["model"]
    encoders = data["encoders"]

    # ROW 1
    st.markdown('<div class="input-card"><div class="card-title">👤 Personal & Physical Profile</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
    with c2: age_cat = st.selectbox("Age Category", ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'], index=7)
    with c3: sex = st.selectbox("Biological Sex", ["Female", "Male"])
    with c4: race = st.selectbox("Race", ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'])
    st.markdown('</div>', unsafe_allow_html=True)

    # ROW 2
    st.markdown('<div class="input-card"><div class="card-title">🏥 Medical History</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: stroke = st.selectbox("History of Stroke?", ["No", "Yes"])
    with c2: diabetic = st.selectbox("Diabetic?", ['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)'])
    with c3: asthma = st.selectbox("Asthma?", ["No", "Yes"])
    with c4: kidney = st.selectbox("Kidney Disease?", ["No", "Yes"])
    
    c5, c6 = st.columns(2)
    with c5: skin = st.selectbox("History of Skin Cancer?", ["No", "Yes"])
    with c6: gen_health = st.selectbox("General Health Self-Eval", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], index=2)
    st.markdown('</div>', unsafe_allow_html=True)

    # ROW 3
    st.markdown('<div class="input-card"><div class="card-title">🏃 Habits & Vitals</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: smoking = st.selectbox("Smoked >100 cigs in life?", ["No", "Yes"])
    with c2: alcohol = st.selectbox("Heavy Alcohol Drinker?", ["No", "Yes"])
    with c3: diff_walk = st.selectbox("Difficulty Walking?", ["No", "Yes"])
    with c4: activity = st.selectbox("Physical Activity (Last 30 days)", ["Yes", "No"])
    
    c5, c6, c7 = st.columns(3)
    with c5: sleep = st.slider("Avg. Sleep Hours", 0.0, 24.0, 7.0, 0.5)
    with c6: phy_health = st.slider("Days Physical Health Bad (Last 30)", 0, 30, 0)
    with c7: ment_health = st.slider("Days Mental Health Bad (Last 30)", 0, 30, 0)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Analyze Risk Profile 🔍"):
        input_dict = {
            "BMI": bmi, "Smoking": smoking, "AlcoholDrinking": alcohol, "Stroke": stroke,
            "PhysicalHealth": float(phy_health), "MentalHealth": float(ment_health),
            "DiffWalking": diff_walk, "Sex": sex, "AgeCategory": age_cat,
            "Race": race, "Diabetic": diabetic, "PhysicalActivity": activity,
            "GenHealth": gen_health, "SleepTime": sleep, "Asthma": asthma,
            "KidneyDisease": kidney, "SkinCancer": skin
        }
        
        feature_order = [
            "BMI", "Smoking", "AlcoholDrinking", "Stroke", "PhysicalHealth", "MentalHealth", "DiffWalking", 
            "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "GenHealth", "SleepTime", 
            "Asthma", "KidneyDisease", "SkinCancer"
        ]
        
        encoded_input = []
        for feature in feature_order:
            val = input_dict[feature]
            if feature in encoders:
                val = encoders[feature].transform([val])[0]
            encoded_input.append(val)

        prediction_prob = model.predict_proba([encoded_input])
        prediction = model.predict([encoded_input])
        risk_percent = prediction_prob[0][1] * 100
        result_str = encoders["HeartDisease"].inverse_transform(prediction)[0]

        st.markdown("---")
        if result_str == "Yes":
            st.markdown(f'<div class="result-box-danger"><h2>⚠️ High Risk Detected</h2><p>Risk Probability: {risk_percent:.1f}%</p></div>', unsafe_allow_html=True)
        else:
            st.balloons()
            st.markdown(f'<div class="result-box-safe"><h2>✅ Low Risk Profile</h2><p>Risk Probability: {risk_percent:.1f}%</p></div>', unsafe_allow_html=True)
else:
    st.error("Application failed to load model data.")
