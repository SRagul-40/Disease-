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
# 1. PAGE CONFIGURATION & BLACK/RED STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* 1. Global Font & Colors */
    .stApp {
        background-color: #ffffff; /* White Background */
        color: #000000; /* Force Black Text Globally */
        font-family: 'Arial', sans-serif;
    }
    
    /* 2. Text Overrides to ensure Black */
    p, div, label, span, .stMarkdown {
        color: #000000 !important;
    }

    /* 3. Headers - RED */
    h1, h2, h3, h4, .main-header {
        color: #b30000 !important; /* Deep Red */
        font-weight: 900 !important;
        text-transform: uppercase;
    }
    
    /* 4. Sub-headers - BLACK */
    .sub-header {
        font-size: 1.3rem;
        color: #000000 !important;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 2px solid #b30000;
        padding-bottom: 10px;
    }

    /* 5. Input Cards */
    .input-card {
        background-color: #ffffff;
        padding: 20px;
        border: 2px solid #000000; /* Black Border */
        border-radius: 0px; /* Sharp corners for stark look */
        box-shadow: 5px 5px 0px #b30000; /* Red Shadow */
        margin-bottom: 20px;
    }
    
    .card-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #b30000 !important; /* Red Title */
        margin-bottom: 15px;
        border-bottom: 2px solid #000000;
        padding-bottom: 5px;
    }

    /* 6. Buttons */
    .stButton>button {
        width: 100%;
        background-color: #b30000 !important; /* Red Background */
        color: #ffffff !important; /* White Text for contrast */
        border: 2px solid #000000 !important;
        font-weight: 800;
        font-size: 1.2rem;
        text-transform: uppercase;
        border-radius: 0px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #000000 !important; /* Black on Hover */
        color: #b30000 !important; /* Red Text on Hover */
        border: 2px solid #b30000 !important;
    }

    /* 7. Result Box - SAFE (Black styling) */
    .result-box-safe {
        background-color: #ffffff;
        color: #000000 !important;
        padding: 30px;
        border: 4px solid #000000;
        text-align: center;
        box-shadow: 10px 10px 0px #cccccc;
    }

    /* 8. Result Box - DANGER (Red styling) */
    .result-box-danger {
        background-color: #ffffff;
        color: #b30000 !important;
        padding: 30px;
        border: 4px solid #b30000;
        text-align: center;
        box-shadow: 10px 10px 0px #000000;
    }
    
    /* 9. Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #f0f0f0;
        border-right: 2px solid #000000;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SMART MODEL LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_data():
    file_path = "heart_disease_2020_model.pkl"
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        status = st.empty()
        status.error("⚙️ Model not found. Training now... Please wait.")
        
        try:
            import kagglehub
            path = kagglehub.dataset_download("aqleemkhan/heart-disease-2020")
            csv_path = os.path.join(path, "heart_2020_cleaned.csv")
            
            if not os.path.exists(csv_path):
                for f in os.listdir(path):
                    if f.endswith(".csv"):
                        csv_path = os.path.join(path, f)
                        break
            
            dia = pd.read_csv(csv_path)
            
            LE = {}
            for col in dia.columns:
                if dia[col].dtype == 'object':
                    LE[col] = LabelEncoder()
                    dia[col] = LE[col].fit_transform(dia[col])
            
            ind = dia.drop('HeartDisease', axis=1)
            dep = dia['HeartDisease']
            X_train, _, y_train, _ = train_test_split(ind, dep, test_size=0.2, random_state=42)
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            data = {"model": model, "encoders": LE}
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            
            status.empty()
            return data
            
        except Exception as e:
            status.error(f"Error: {e}")
            return None

data = get_model_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h1>CARDIO AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**SYSTEM STATUS:** ONLINE")
    st.markdown("**MODEL:** LOGISTIC REGRESSION")
    st.markdown("---")
    st.markdown("### ⚠️ DISCLAIMER")
    st.markdown("This tool is for educational purposes only.")

# -----------------------------------------------------------------------------
# 4. MAIN INTERFACE
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">HEART DISEASE ASSESSMENT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ENTER CLINICAL DATA FOR ANALYSIS</div>', unsafe_allow_html=True)

if data:
    model = data["model"]
    encoders = data["encoders"]

    # ROW 1
    st.markdown('<div class="input-card"><div class="card-title">1. PERSONAL DATA</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
    with c2: age_cat = st.selectbox("Age Category", ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'], index=7)
    with c3: sex = st.selectbox("Sex", ["Female", "Male"])
    with c4: race = st.selectbox("Race", ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'])
    st.markdown('</div>', unsafe_allow_html=True)

    # ROW 2
    st.markdown('<div class="input-card"><div class="card-title">2. MEDICAL HISTORY</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: stroke = st.selectbox("History of Stroke?", ["No", "Yes"])
    with c2: diabetic = st.selectbox("Diabetic?", ['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)'])
    with c3: asthma = st.selectbox("Asthma?", ["No", "Yes"])
    with c4: kidney = st.selectbox("Kidney Disease?", ["No", "Yes"])
    
    c5, c6 = st.columns(2)
    with c5: skin = st.selectbox("History of Skin Cancer?", ["No", "Yes"])
    with c6: gen_health = st.selectbox("Self-Reported Health", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], index=2)
    st.markdown('</div>', unsafe_allow_html=True)

    # ROW 3
    st.markdown('<div class="input-card"><div class="card-title">3. HABITS & VITALS</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: smoking = st.selectbox("Smoker?", ["No", "Yes"])
    with c2: alcohol = st.selectbox("Heavy Alcohol Use?", ["No", "Yes"])
    with c3: diff_walk = st.selectbox("Difficulty Walking?", ["No", "Yes"])
    with c4: activity = st.selectbox("Physical Activity (30 days)", ["Yes", "No"])
    
    c5, c6, c7 = st.columns(3)
    with c5: sleep = st.slider("Sleep (Hours)", 0.0, 24.0, 7.0, 0.5)
    with c6: phy_health = st.slider("Bad Physical Health Days", 0, 30, 0)
    with c7: ment_health = st.slider("Bad Mental Health Days", 0, 30, 0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("RUN ANALYSIS"):
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
            st.markdown(
                f"""
                <div class="result-box-danger">
                    <h1>⚠️ HIGH RISK DETECTED</h1>
                    <hr style="border: 2px solid #b30000">
                    <h2 style="color: #b30000 !important;">PROBABILITY: {risk_percent:.1f}%</h2>
                    <p style="font-weight: bold;">CONSULT A MEDICAL PROFESSIONAL IMMEDIATELY.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-box-safe">
                    <h1>✅ LOW RISK PROFILE</h1>
                    <hr style="border: 2px solid #000000">
                    <h2 style="color: #000000 !important;">PROBABILITY: {risk_percent:.1f}%</h2>
                    <p style="font-weight: bold;">MAINTAIN HEALTHY HABITS.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
else:
    st.error("System Error: Model failed to load.")
