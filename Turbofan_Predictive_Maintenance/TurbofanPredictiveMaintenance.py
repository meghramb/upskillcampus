import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Turbofan Engine RUL Predictor", page_icon="✈️", layout="wide")

st.title("✈️ Jet Engine Predictive Maintenance")
st.markdown("Upload engine sensor data to predict Remaining Useful Life (RUL) and prevent failures.")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('turbofan_xgb_model.pkl')

model = load_model()

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test_FD001.txt", type=["txt"])

if uploaded_file is not None:
    # Read Data
    columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=columns)
    
    st.success("✅ Data loaded successfully!")
    
    # Select Engine Number
    engine_ids = df['unit_number'].unique()
    selected_engine = st.selectbox("Select Engine ID for Prediction:", engine_ids)
    
    # Filter data for selected engine
    engine_data = df[df['unit_number'] == selected_engine]
    
    # Constant columns to drop (Same as training)
    drop_cols = ['setting_1', 'setting_2', 'setting_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    engine_features = engine_data.drop(columns=drop_cols)
    features = [col for col in engine_features.columns if col.startswith('sensor')]
    
    # Get the last cycle data (Current state of engine)
    current_state = engine_features[features].iloc[-1:]
    
    # Predict RUL
    predicted_rul = model.predict(current_state)[0]
    
    # Display Result elegantly
    st.subheader(f"Prediction for Engine #{selected_engine}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Remaining Useful Life (Cycles)", value=f"{int(predicted_rul)}")
        
        if predicted_rul < 30:
            st.error("🚨 CRITICAL: Engine requires immediate maintenance!")
        elif predicted_rul < 60:
            st.warning("⚠️ WARNING: Schedule maintenance soon.")
        else:
            st.success("✅ HEALTHY: Engine is operating normally.")
            
    with col2:
        # Plot sensor degradation trend (e.g., Sensor 14)
        st.write("Sensor 14 Degradation Trend")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(engine_data['time_in_cycles'], engine_data['sensor_14'], color='orange')
        ax.set_xlabel("Time (Cycles)")
        ax.set_ylabel("Sensor 14 Value")
        st.pyplot(fig)

else:
    st.info("Please upload the 'test_FD001.txt' dataset from the sidebar to continue.")