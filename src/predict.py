
from pathlib import Path
import joblib
import pandas as pd

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the trained model
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model_v1.joblib"
model = joblib.load(MODEL_PATH)


def predict_house_price(
    area,
    bedrooms,
    bathrooms,
    age,
    location,
    property_type
):
    """Predict the price of a house from its features."""

    house = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": location,
        "Property_Type": property_type
    }])

    predicted_price = model.predict(house)[0]
    return predicted_price


if __name__ == "__main__":
    price = predict_house_price(
        area=2000,
        bedrooms=3,
        bathrooms=2,
        age=10,
        location="City Center",
        property_type="Apartment"
    )

    print(f"Predicted house price: ₹{price:,.2f}")
