import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Function to convert time string to m/s
def calculate_single_speed(time_str, distance):
    try:
        if ':' in str(time_str):
            minutes, seconds = str(time_str).split(':')
            total_seconds = int(minutes) * 60 + float(seconds)
        else:
            total_seconds = float(time_str)
        return distance / total_seconds if total_seconds > 0 else 0
    except:
        return 0

st.set_page_config(page_title="HKJC Prediction", layout="wide")
st.title("🏇 HKJC Horse Racing Results Predictor")

model_path = 'random_forest_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    st.subheader("1. Current Race Information")
    curr_col1, curr_col2 = st.columns(2)
    with curr_col1:
        distance = st.selectbox("Current Race Distance (m)", [1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400])
        actual_weight = st.number_input("Horse's Weight (lbs)", value=120.0)
    with curr_col2:
        days_since = st.number_input("Days Since Last Run", value=14)
        barrier = st.slider("Barrier (Draw)", 1, 14, 7)
        class_move = st.selectbox("Class Change", options=[-1, 0, 1], format_func=lambda x: "Down" if x==-1 else ("Up" if x==1 else "Same"))

    st.markdown("---")
    st.subheader("2. Information on Results of Past 3 Runs")
    
    # Create three columns for the three past races
    p1, p2, p3 = st.columns(3)
    
    with p1:
        st.write("**Race 1 (Most Recent)**")
        t1 = st.text_input("Finishing Time (e.g. 1:10.2)", value="1:09.8", key="t1")
        d1 = st.selectbox("Race Distance (m)", [1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400], index=1, key="d1")
        res1 = st.checkbox("Finished Top 3?", key="r1")
        
    with p2:
        st.write("**Race 2**")
        t2 = st.text_input("Finishing Time (e.g. 1:10.5)", value="1:10.1", key="t2")
        d2 = st.selectbox("Race Distance (m)", [1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400], index=1, key="d2")
        res2 = st.checkbox("Finished Top 3?", key="r2")
        
    with p3:
        st.write("**Race 3**")
        t3 = st.text_input("Finishing Time (e.g. 1:09.9)", value="1:10.4", key="t3")
        d3 = st.selectbox("Race Distance (m)", [1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400], index=1, key="d3")
        res3 = st.checkbox("Finished Top 3?", key="r3")

    if st.button("Predict Result"):
        # --- FEATURE ENGINEERING ---
        
        # 1. Calculate Individual Speeds
        s1 = calculate_single_speed(t1, d1)
        s2 = calculate_single_speed(t2, d2)
        s3 = calculate_single_speed(t3, d3)
        
        # 2. Hist_Speed_Rating (Average of the three)
        avg_speed = (s1 + s2 + s3) / 3.0
        
        # 3. P3R_Top3_Pct_x_Dist
        top3_count = sum([res1, res2, res3])
        calc_p3r_dist = (top3_count / 3.0) * distance
        
        # 4. Rel_Weight_x_Dist (using 120 as a baseline)
        calc_rel_weight_dist = (actual_weight / 120.0) * distance

        # --- PREDICTION ---
        data = {
            'P3R_Top3_Pct_x_Dist': calc_p3r_dist,
            'Days_Since_Last_Run': days_since,
            'Rel_Weight_x_Dist': calc_rel_weight_dist,
            'Barrier_Rank': barrier,
            'Class_Change': class_move,
            'Hist_Speed_Rating': avg_speed
        }
        
        feature_order = ['P3R_Top3_Pct_x_Dist', 'Days_Since_Last_Run', 'Rel_Weight_x_Dist', 
                         'Barrier_Rank', 'Class_Change', 'Hist_Speed_Rating']
        
        input_df = pd.DataFrame([data])[feature_order]
        
        st.write(f"**Engineered Avg Speed:** {avg_speed:.2f} m/s")
        
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)[0][1]
        
        if prediction[0] == 1:
            st.success(f"PROBABLE TOP 3 (Confidence: {prob:.2%})")
        else:
            st.warning(f"OUTSIDE TOP 3 (Confidence: {1-prob:.2%})")
else:
    st.error("Model file not found!")
