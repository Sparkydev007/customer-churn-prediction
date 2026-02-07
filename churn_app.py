
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .churn { background-color: #ffebee; color: #c62828; }
    .no-churn { background-color: #e8f5e9; color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("customer_churn_model.pkl")
        encoders = joblib.load("churn_encoders.pkl")
        return model, encoders
    except FileNotFoundError:
        st.error("Model files not found. Run the notebook first.")
        return None, None

model, encoders = load_model()

st.title(" Customer Churn Prediction")
st.markdown("Predict customer churn with Machine Learning")
st.markdown("---")

if model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    
    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
                                      ["Electronic check", "Mailed check", 
                                       "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0, 5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure, 50.0)
    
    st.markdown("---")
    
    if st.button("Predict Churn"):
        input_data = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner,
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
            'MultipleLines': multiple_lines, 'InternetService': internet_service,
            'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode binary features
        input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            if col in input_df.columns:
                input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})
        
        # Encode categorical features using saved encoders
        for column, encoder in encoders.items():
            if column in input_df.columns and input_df[column].dtype == 'object':
                input_df[column] = encoder.transform(input_df[column])
        
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box churn">WILL CHURN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box no-churn">WILL STAY</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{max(proba) * 100:.1f}%")
        
        with col3:
            risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
            st.metric("Risk Level", risk)
        
        fig = go.Figure(data=[
            go.Bar(name='No Churn', x=['Probability'], y=[proba[0]], marker_color='#2ecc71'),
            go.Bar(name='Churn', x=['Probability'], y=[proba[1]], marker_color='#e74c3c')
        ])
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if prediction == 1:
            st.warning("**At Risk:** Offer retention incentives, upgrade to longer contract, provide better support")
        else:
            st.success("**Low Risk:** Continue excellent service, send surveys, offer loyalty benefits")
    
    with st.sidebar:
        st.header("Model Info")
        st.info("""
        **XGBoost Classifier**
        - Test Accuracy: ~77%
        - Training: 7,043 customers
        - Features: 19 attributes
        - Balanced with SMOTE
        """)
        
        st.header("Top Predictors")
        st.markdown("""
        - Contract type
        - Tenure duration  
        - Monthly charges
        - Internet service
        - Payment method
        """)
        
        st.markdown("---")
        st.caption("Built with Streamlit & Scikit-learn")

else:
    st.error("Model not found. Run the notebook to generate model files.")
