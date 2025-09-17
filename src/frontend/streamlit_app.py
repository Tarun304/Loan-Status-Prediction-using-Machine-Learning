import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Loan Predictor", 
    page_icon="🏦",
    layout="centered"
)

# Initialize session state for reset functionality
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        border: none;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background-color: #c0392b;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🏦 Loan Status Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

API_URL = "https://loan-prediction-api-1hq4.onrender.com"

# Personal Information Section
st.markdown('<h2 class="section-header">👤 Personal Information</h2>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", 0, 10, 0, key=f"dependents_{st.session_state.reset_key}")
    education = st.selectbox("Education Level", ["8th", "10th", "12th", "Graduate"], key=f"education_{st.session_state.reset_key}")

with col2:
    self_employed = st.selectbox("Self Employed", ["No", "Yes"], key=f"self_employed_{st.session_state.reset_key}")
    employment_type = st.selectbox("Employment Type", ["Salaried", "Business", "Freelancer"], key=f"employment_{st.session_state.reset_key}")

# Financial Information Section
st.markdown('<h2 class="section-header">💰 Financial Information</h2>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    income_annum = st.number_input("Annual Income (₹)", 0.0, step=10000.0, format="%.0f", key=f"income_{st.session_state.reset_key}")
    st.caption("💡 Enter your total annual income before taxes")
    loan_amount = st.number_input("Loan Amount (₹)", 0.0, step=50000.0, format="%.0f", key=f"loan_amount_{st.session_state.reset_key}")
    st.caption("💡 Amount you want to borrow")

with col4:
    loan_term = st.number_input("Loan Term (years)", min_value=0.1, step=0.5, value=10.0, key=f"loan_term_{st.session_state.reset_key}")
    cibil_score = st.number_input("CIBIL Score", 300, 900, 700, key=f"cibil_{st.session_state.reset_key}")

# Assets Information Section
st.markdown('<h2 class="section-header">🏠 Assets Information</h2>', unsafe_allow_html=True)
col5, col6 = st.columns(2)

with col5:
    residential_assets_value = st.number_input("Residential Assets (₹)", 0.0, step=100000.0, format="%.0f", key=f"residential_{st.session_state.reset_key}")
    commercial_assets_value = st.number_input("Commercial Assets (₹)", 0.0, step=50000.0, format="%.0f", key=f"commercial_{st.session_state.reset_key}")

with col6:
    luxury_assets_value = st.number_input("Luxury Assets (₹)", 0.0, step=25000.0, format="%.0f", key=f"luxury_{st.session_state.reset_key}")
    bank_asset_value = st.number_input("Bank Assets (₹)", 0.0, step=10000.0, format="%.0f", key=f"bank_{st.session_state.reset_key}")

# Prediction Section
st.markdown("---")

# Add predict and reset buttons
col_predict, col_reset = st.columns([3, 1])

with col_predict:
    predict_clicked = st.button("🔍 Predict Loan Status", type="primary")

with col_reset:
    if st.button("🔄 Reset"):
        # Increment reset key to force all widgets to reset
        st.session_state.reset_key += 1
        st.rerun()

# Rest of your prediction logic stays exactly the same...
if predict_clicked:
    # Your existing validation and prediction code here
    validation_errors = []
    
    if income_annum <= 0:
        validation_errors.append("Annual income must be greater than 0")
    
    if loan_amount <= 0:
        validation_errors.append("Loan amount must be greater than 0")
    
    if loan_amount > income_annum * 10:
        validation_errors.append("Loan amount seems too high compared to annual income (max 10x income)")
    
    if cibil_score < 300:
        validation_errors.append("CIBIL score cannot be less than 300")
    
    if self_employed == "Yes" and employment_type == "Salaried":
        validation_errors.append("Self-employed person cannot have Salaried employment type")
    
    if self_employed == "No" and employment_type in ["Business", "Freelancer"]:
        validation_errors.append("Non self-employed person cannot have Business/Freelancer employment type")
    
    if validation_errors:
        for error in validation_errors:
            st.error(f"⚠️ {error}")
    else:
        data = {
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "employment_type": employment_type,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }
        
        with st.spinner("🤖 Our AI is analyzing your application... This may take a few seconds"):
            try:
                response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    
                    if result["status"] == "Approved":
                        st.markdown(f"""
                        <div style="
                            padding: 0.75rem 1rem;
                            border-radius: 6px;
                            margin: 1rem 0;
                            background-color: #d1f2eb;
                            border-left: 4px solid #27ae60;
                            text-align: left;
                        ">
                            <span style="
                                color: #27ae60;
                                font-size: 1.1rem;
                                font-weight: 600;
                            ">✅ Loan Status: APPROVED</span>
                            <br>
                            <span style="
                                font-size: 0.85rem;
                                color: #7f8c8d;
                                font-style: italic;
                            ">Confidence: {result['probability']:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            padding: 0.75rem 1rem;
                            border-radius: 6px;
                            margin: 1rem 0;
                            background-color: #fdf2f2;
                            border-left: 4px solid #e74c3c;
                            text-align: left;
                        ">
                            <span style="
                                color: #e74c3c;
                                font-size: 1.1rem;
                                font-weight: 600;
                            ">❌ Loan Status: REJECTED</span>
                            <br>
                            <span style="
                                font-size: 0.85rem;
                                color: #7f8c8d;
                                font-style: italic;
                            ">Confidence: {result['probability']:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.expander("📋 View Application Summary"):
                        st.json(data)
                        
                elif response.status_code == 422:
                    st.error("⚠️ Invalid input data. Please check your values.")
                elif response.status_code == 500:
                    st.error("⚠️ Server error. Please try again later.")
                else:
                    st.error(f"⚠️ API Error: Unable to process request (Status {response.status_code})")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API. Make sure the backend server is running.")
            except Exception as e:
                st.error(f"❗ Unexpected error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">Built with ❤️ using Streamlit & FastAPI</p>',
    unsafe_allow_html=True
)
