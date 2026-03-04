import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="CarDekho Price Predictor",
    page_icon="🚗",
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="expanded"
)

# --- 2. Load the Model and Artifacts (Cached) ---
@st.cache_resource
def load_artifacts():
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    return model, encoders, model_columns

try:
    model, encoders, model_columns = load_artifacts()
except FileNotFoundError:
    st.error("🚨 Error: Model files not found. Please ensure 'car_price_model.pkl', 'label_encoders.pkl', and 'model_columns.pkl' are in the same directory.")
    st.stop()

# --- 3. App Header & Banner ---
# Optional: Add a banner image. 
# Make sure you have an image named 'car_banner.jpg' in the same folder.
st.image("car_banner.jpg", use_container_width=True)

st.title("🚗 Used Car Price Predictor")
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<p class="big-font">Get an instant, AI-powered estimate for your car\'s market value.</p>', unsafe_allow_html=True)
st.write("---")

# --- 4. Sidebar for Model Selection & About ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Adjust the car's core details here.")
    
    # Get the list of brands and sort them
    brand_options = sorted(encoders['oem'].classes_)
    selected_brand = st.selectbox("Select Brand (OEM)", brand_options)
    
    # Filter models based on the selected brand
    # (This assumes your encoder classes are in a format like 'Brand Model')
    # A simpler way for now is to show all models, or you can improve this logic.
    model_options = sorted([m for m in encoders['model'].classes_ if selected_brand.lower() in m.lower()])
    if not model_options: # Fallback if filtering fails
        model_options = sorted(encoders['model'].classes_)

    selected_model = st.selectbox("Select Model", model_options)
    
    st.write("---")
    st.markdown("### ℹ️ About")
    st.info("This tool uses a machine learning model trained on thousands of real-world car listings to estimate market price based on vehicle specifications.")
    st.markdown("[View the Source Code on GitHub](https://github.com/)") # Replace with your link

# --- 5. Main Input Area (Columns) ---
st.subheader("📝 Enter Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    myear = st.slider("Manufacturing Year", min_value=2000, max_value=2024, value=2018, step=1, help="The year the car was manufactured.")
    km = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000, help="Total distance the car has been driven.")

with col2:
    transmission = st.radio("Transmission Type", ["Manual", "Automatic"], horizontal=True)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])

# --- 6. Prediction Logic ---
st.write("---")
predict_button = st.button("🚀 Predict Price", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("Calculating estimate..."):
        # Create a blank dataframe with all our original columns set to 0
        input_data = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Fill in the numerical inputs
        input_data['myear'] = myear
        input_data['km'] = km
        
        # Fill in and encode the high-cardinality text inputs
        input_data['oem'] = encoders['oem'].transform([selected_brand])[0]
        input_data['model'] = encoders['model'].transform([selected_model])[0]
        
        # Handle the One-Hot Encoded columns
        trans_col = f"transmission_{transmission.lower()}"
        if trans_col in input_data.columns:
            input_data[trans_col] = 1
            
        fuel_col = f"fuel_{fuel.lower()}"
        if fuel_col in input_data.columns:
            input_data[fuel_col] = 1

        # Predict and format the output
        try:
            # Predict the log price, then use expm1 to convert it back to Rupees
            log_price = model.predict(input_data)
            final_price = np.expm1(log_price)[0]
            
            # Display the result with a nice visual
            st.balloons()
            st.success(f"### Estimated Market Value:")
            st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>₹ {final_price:,.0f}</h1>", unsafe_allow_html=True)
            st.info("Note: This is an estimate. The actual selling price may vary based on car condition, location, and market demand.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- 7. Footer ---
st.write("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with ❤️ using Streamlit & XGBoost</p>", unsafe_allow_html=True)