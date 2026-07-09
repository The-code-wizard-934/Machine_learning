from pathlib import Path
import datetime as dt

import pandas as pd
import streamlit as st
import xgboost as xgb


MODEL_PATH = Path(r"G:\Machine\FINALISING\xgb_model.json")


@st.cache_resource
def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def encode_fuel(fuel_type: str) -> int:
    mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    return mapping[fuel_type]


def encode_seller(seller_type: str) -> int:
    mapping = {"Individual": 0, "Dealer": 1}
    return mapping[seller_type]


def encode_transmission(transmission: str) -> int:
    mapping = {"Manual": 0, "Automatic": 1}
    return mapping[transmission]


def main() -> None:
    st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

    st.title("Car Price Prediction")
    st.write("Enter the car details below to estimate its selling price using the trained XGBoost model.")

    model = load_model()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input("Year of Manufacture", min_value=1990, max_value=2026, value=2015, step=1)
            present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0, step=0.1, format="%.2f")
            kms_driven = st.number_input("Kms Driven", min_value=0, value=30000, step=500)

        with col2:
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

        owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
        submitted = st.form_submit_button("Predict Selling Price")

    if submitted:
        age = dt.date.today().year - int(year)
        input_frame = pd.DataFrame(
            [[
                float(present_price),
                int(kms_driven),
                encode_fuel(fuel_type),
                encode_seller(seller_type),
                encode_transmission(transmission),
                int(owner),
                int(age),
            ]],
            columns=["Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner", "Age"],
        )

        prediction = float(model.predict(input_frame)[0])

        st.success(f"Estimated selling price: {prediction:.2f} lakhs")
        st.caption("The prediction uses the training schema from the saved model in this folder.")


if __name__ == "__main__":
    main()
    