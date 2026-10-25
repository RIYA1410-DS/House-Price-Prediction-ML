
from pathlib import Path
import sys

import streamlit as st

# Allow imports from the src folder
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_house_price

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

st.title("🏡 House Price Prediction")
st.write("Enter property details to estimate its price.")

with st.form("house_prediction_form"):
    area = st.number_input(
        "Area (square feet)",
        min_value=1,
        max_value=100000,
        value=2000
    )

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=20,
            value=3
        )

    with col2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0,
            max_value=20,
            value=2
        )

    age = st.number_input(
        "Property age (years)",
        min_value=0,
        max_value=500,
        value=10
    )

    location = st.selectbox(
        "Location",
        ["City Center", "Suburb", "Rural"]
    )

    property_type = st.selectbox(
        "Property type",
        ["Apartment", "House", "Villa"]
    )

    submitted = st.form_submit_button("Predict House Price")

if submitted:
    try:
        price = predict_house_price(
            area=area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            age=age,
            location=location,
            property_type=property_type
        )

        st.success("Prediction completed!")
        st.metric("Estimated House Price", f"₹{price:,.2f}")
        st.caption(
            "This is a model-based estimate, not a guaranteed market price."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
