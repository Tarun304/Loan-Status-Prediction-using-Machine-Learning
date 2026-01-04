"""Streamlit UI for Loan Status Prediction"""

import streamlit as st
from utils import API_URL, FEATURE_INFO, check_api_health, predict_via_api

# Page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .approved-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .rejected-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        font-size: 1.1rem;
    }
    .info-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<p class="main-header">💰 Loan Approval Predictor</p>', unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray; margin-bottom: 2rem;'>AI-Powered Loan Approval Prediction System</p>",
    unsafe_allow_html=True,
)

# Check API health
with st.spinner("🔄 Connecting to API..."):
    api_healthy = check_api_health()

if not api_healthy:
    st.error(
        f"❌ Cannot connect to API at {API_URL}. Please make sure FastAPI is running!"
    )
    st.info("💡 Run FastAPI with: `uvicorn src.api.app:app --reload`")
    st.stop()
else:
    st.success(f"✅ Connected to API at {API_URL}")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Predict Loan Status", "📖 Feature Guide", "📊 About"])

# TAB 1: Prediction
with tab1:
    st.subheader("Enter Loan Application Details")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        features = {}

        # Left column - Personal & Employment Info
        with col1:
            st.markdown("### 👤 Personal Information")

            info = FEATURE_INFO["no_of_dependents"]
            features["no_of_dependents"] = st.number_input(
                info["name"],
                min_value=info["range"][0],
                max_value=info["range"][1],
                value=info["default"],
                help=info["desc"],
            )

            info = FEATURE_INFO["education"]
            features["education"] = st.selectbox(
                info["name"], options=info["options"], help=info["desc"]
            )

            info = FEATURE_INFO["self_employed"]
            features["self_employed"] = st.selectbox(
                info["name"], options=info["options"], help=info["desc"]
            )

            info = FEATURE_INFO["employment_type"]
            features["employment_type"] = st.selectbox(
                info["name"], options=info["options"], help=info["desc"]
            )

            st.markdown("### 💵 Financial Information")

            info = FEATURE_INFO["income_annum"]
            features["income_annum"] = st.number_input(
                info["name"],
                min_value=info["range"][0],
                max_value=info["range"][1],
                value=info["default"],
                step=100000.0,
                help=info["desc"],
                format="%.2f",
            )

            info = FEATURE_INFO["cibil_score"]
            features["cibil_score"] = st.slider(
                info["name"],
                min_value=int(info["range"][0]),
                max_value=int(info["range"][1]),
                value=int(info["default"]),
                help=info["desc"],
            )

        # Right column - Loan & Assets Info
        with col2:
            st.markdown("### 🏦 Loan Details")

            info = FEATURE_INFO["loan_amount"]
            features["loan_amount"] = st.number_input(
                info["name"],
                min_value=info["range"][0],
                max_value=info["range"][1],
                value=info["default"],
                step=100000.0,
                help=info["desc"],
                format="%.2f",
            )

            info = FEATURE_INFO["loan_term"]
            features["loan_term"] = st.number_input(
                info["name"],
                min_value=info["range"][0],
                max_value=info["range"][1],
                value=info["default"],
                step=1.0,
                help=info["desc"],
                format="%.1f",
            )

            st.markdown("### 🏠 Assets Information")

            info = FEATURE_INFO["residential_assets_value"]
            features["residential_assets_value"] = st.number_input(
                info["name"],
                min_value=info["range"][0],
                max_value=info["range"][1],
                value=info["default"],
                step=100000.0,
                help=info["desc"],
                format="%.2f",
            )

            info = FEATURE_INFO["commercial_assets_value"]
            features["commercial_assets_value"] = st.number_input(
                info["name"],
                min_value=info["range"][0],
                max_value=info["range"][1],
                value=info["default"],
                step=100000.0,
                help=info["desc"],
                format="%.2f",
            )

            info = FEATURE_INFO["luxury_assets_value"]
            features["luxury_assets_value"] = st.number_input(
                info["name"],
                min_value=info["range"][0],
                max_value=info["range"][1],
                value=info["default"],
                step=100000.0,
                help=info["desc"],
                format="%.2f",
            )

            info = FEATURE_INFO["bank_asset_value"]
            features["bank_asset_value"] = st.number_input(
                info["name"],
                min_value=info["range"][0],
                max_value=info["range"][1],
                value=info["default"],
                step=100000.0,
                help=info["desc"],
                format="%.2f",
            )

        submit_button = st.form_submit_button(
            "🔮 Predict Loan Status", use_container_width=True
        )

    if submit_button:
        with st.spinner("Analyzing loan application..."):
            try:
                result = predict_via_api(features)

                status = result["status"]
                probability = result["probability"]
                model_version = result["model_version"]

                if status == "Approved":
                    st.markdown('<div class="approved-box">', unsafe_allow_html=True)
                    st.markdown("### ✅ Loan Status: APPROVED")
                    st.markdown(f"# {probability*100:.2f}% Confidence")
                    st.markdown(f"*Model Version: {model_version}*")
                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.markdown('<div class="rejected-box">', unsafe_allow_html=True)
                    st.markdown("### ❌ Loan Status: REJECTED")
                    st.markdown(f"# {(1-probability)*100:.2f}% Confidence")
                    st.markdown(f"*Model Version: {model_version}*")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Show feature summary
                with st.expander("📋 Application Summary"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Annual Income", f"₹{features['income_annum']:,.0f}")
                        st.metric("CIBIL Score", f"{features['cibil_score']}")
                    with col2:
                        st.metric("Loan Amount", f"₹{features['loan_amount']:,.0f}")
                        st.metric("Loan Term", f"{features['loan_term']} years")
                    with col3:
                        total_assets = (
                            features["residential_assets_value"]
                            + features["commercial_assets_value"]
                            + features["luxury_assets_value"]
                            + features["bank_asset_value"]
                        )
                        st.metric("Total Assets", f"₹{total_assets:,.0f}")
                        st.metric("Employment", features["employment_type"])

            except Exception as e:
                st.error(f"❌ {str(e)}")

# TAB 2: Feature Guide
with tab2:
    st.subheader("📖 Feature Descriptions")
    st.write("Learn more about each input feature used by the model:")
    st.markdown("---")

    categories = {
        "👤 Personal Information": [
            "no_of_dependents",
            "education",
            "self_employed",
            "employment_type",
        ],
        "💵 Financial Information": ["income_annum", "cibil_score"],
        "🏦 Loan Details": ["loan_amount", "loan_term"],
        "🏠 Assets Information": [
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ],
    }

    for category, feature_list in categories.items():
        st.markdown(f"### {category}")
        for feature in feature_list:
            info = FEATURE_INFO[feature]
            with st.expander(f"**{info['name']}**"):
                st.write(f"**Description:** {info['desc']}")
                if info["type"] == "select":
                    st.write(f"**Options:** {', '.join(info['options'])}")
                else:
                    st.write(
                        f"**Valid Range:** {info['range'][0]:,.0f} - {info['range'][1]:,.0f}"
                    )
                st.write(f"**Default Value:** {info['default']}")
        st.markdown("---")

# TAB 3: About
with tab3:
    st.subheader("📊 About This Application")

    st.markdown(
        """
    ### Overview
    This application uses Machine Learning to predict loan approval status based on applicant information.
    The model analyzes multiple factors including:
    
    - Personal and employment details
    - Financial stability indicators
    - Credit history (CIBIL score)
    - Asset holdings
    - Loan characteristics
    
    ### Model Information
    - **Algorithm:** XGBoost Classifier
    - **Features Used:** 6 carefully selected features
    - **Accuracy:** >85% on test data
    - **F1 Score:** >85%
    
    ### How It Works
    1. Enter your loan application details in the prediction tab
    2. Click "Predict Loan Status"
    3. The model analyzes your information
    4. Receive instant approval/rejection prediction with confidence score
    
    ### Important Notes
    - This is a prediction tool and not a guarantee of loan approval
    - Actual loan decisions depend on many additional factors
    - Use this as a preliminary assessment tool
    
    ### Technology Stack
    - **Frontend:** Streamlit
    - **Backend API:** FastAPI
    - **ML Framework:** XGBoost, Scikit-learn
    - **MLOps:** MLflow, DVC
    - **Deployment:** Docker
    """
    )

    st.markdown("---")
    st.info(
        "💡 **Tip:** Higher CIBIL scores and stable employment significantly improve approval chances!"
    )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Built by Tarun Kumar Behera | MLOps Production Pipeline"
    "</p>",
    unsafe_allow_html=True,
)
