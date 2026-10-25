
import math

from src.predict import predict_house_price


def test_prediction_returns_a_number():
    price = predict_house_price(
        area=2000,
        bedrooms=3,
        bathrooms=2,
        age=10,
        location="City Center",
        property_type="Apartment"
    )

    assert isinstance(price, (int, float))
    assert math.isfinite(price)
    assert price > 0


def test_smaller_property_prediction():
    price = predict_house_price(
        area=1000,
        bedrooms=2,
        bathrooms=1,
        age=20,
        location="Suburb",
        property_type="Apartment"
    )

    assert math.isfinite(price)
    assert price > 0
