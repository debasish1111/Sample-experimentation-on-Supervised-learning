"""
Car Price Predictor — Streamlit app
Loads car_price_model.pkl (a scikit-learn Pipeline) and serves predictions
through a simple web UI.

Run locally with:
    pip install streamlit joblib pandas scikit-learn
    streamlit run streamlit_app.py

Make sure car_price_model.pkl is in the same folder as this script
(or update MODEL_PATH below).
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "car_price_model.pkl"

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🚗 Car Price Predictor")
st.write("Fill in the car's specs and get an estimated price from the trained model.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch (CSV) prediction"])

# ---------------------------------------------------------------------------
# TAB 1: Single car, interactive form
# ---------------------------------------------------------------------------
with tab_single:
    with st.form("car_form"):
        st.subheader("Basic info")
        col1, col2, col3 = st.columns(3)
        with col1:
            brand = st.selectbox(
                "Brand",
                sorted(['alfa-romero', 'audi', 'bmw', 'buick', 'chevrolet', 'dodge',
                        'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi',
                        'nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab',
                        'subaru', 'toyota', 'volkswagen', 'volvo'])
            )
            fueltype = st.selectbox("Fuel type", ["gas", "diesel"])
            aspiration = st.selectbox("Aspiration", ["std", "turbo"])
        with col2:
            doornumber = st.selectbox("Doors", ["two", "four"])
            carbody = st.selectbox(
                "Body style", ["sedan", "hatchback", "wagon", "hardtop", "convertible"]
            )
            drivewheel = st.selectbox("Drive wheel", ["fwd", "rwd", "4wd"])
        with col3:
            enginelocation = st.selectbox("Engine location", ["front", "rear"])
            enginetype = st.selectbox(
                "Engine type", ["ohc", "ohcf", "ohcv", "dohc", "dohcv", "l", "rotor"]
            )
            cylindernumber = st.selectbox(
                "Cylinders", ["two", "three", "four", "five", "six", "eight", "twelve"]
            )

        fuelsystem = st.selectbox(
            "Fuel system", ["mpfi", "2bbl", "idi", "1bbl", "spdi", "4bbl", "mfi", "spfi"]
        )

        st.subheader("Dimensions & weight")
        col4, col5, col6 = st.columns(3)
        with col4:
            wheelbase = st.slider("Wheelbase (in)", 86.0, 121.0, 98.0)
            carlength = st.slider("Length (in)", 141.0, 209.0, 174.0)
        with col5:
            carwidth = st.slider("Width (in)", 60.0, 73.0, 66.0)
            carheight = st.slider("Height (in)", 47.0, 60.0, 54.0)
        with col6:
            curbweight = st.slider("Curb weight (lb)", 1488, 4066, 2556)
            symboling = st.slider("Symboling (risk rating)", -2, 3, 0)

        st.subheader("Engine & performance")
        col7, col8, col9 = st.columns(3)
        with col7:
            enginesize = st.slider("Engine size (cc)", 61, 326, 127)
            horsepower = st.slider("Horsepower", 48, 288, 104)
        with col8:
            boreratio = st.slider("Bore ratio", 2.5, 4.0, 3.3)
            stroke = st.slider("Stroke", 2.0, 4.2, 3.3)
        with col9:
            compressionratio = st.slider("Compression ratio", 7.0, 23.0, 10.1)
            peakrpm = st.slider("Peak RPM", 4150, 6600, 5125)

        st.subheader("Fuel economy")
        col10, col11 = st.columns(2)
        with col10:
            citympg = st.slider("City MPG", 13, 49, 25)
        with col11:
            highwaympg = st.slider("Highway MPG", 16, 54, 31)

        submitted = st.form_submit_button("Predict price")

    if submitted:
        input_df = pd.DataFrame([{
            "symboling": symboling,
            "fueltype": fueltype,
            "aspiration": aspiration,
            "doornumber": doornumber,
            "carbody": carbody,
            "drivewheel": drivewheel,
            "enginelocation": enginelocation,
            "wheelbase": wheelbase,
            "carlength": carlength,
            "carwidth": carwidth,
            "carheight": carheight,
            "curbweight": curbweight,
            "enginetype": enginetype,
            "cylindernumber": cylindernumber,
            "enginesize": enginesize,
            "fuelsystem": fuelsystem,
            "boreratio": boreratio,
            "stroke": stroke,
            "compressionratio": compressionratio,
            "horsepower": horsepower,
            "peakrpm": peakrpm,
            "citympg": citympg,
            "highwaympg": highwaympg,
            "brand": brand,
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"### Estimated price: **${prediction:,.2f}**")

        with st.expander("See the input data sent to the model"):
            st.dataframe(input_df.T.rename(columns={0: "value"}))

# ---------------------------------------------------------------------------
# TAB 2: Batch predictions from an uploaded CSV
# ---------------------------------------------------------------------------
with tab_batch:
    st.write(
        "Upload a CSV with the same columns the model was trained on "
        "(same as `CarPrice_Assignment.csv`, minus `car_ID`, `CarName`, and `price` — "
        "plus a `brand` column instead of `CarName`)."
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        try:
            preds = model.predict(batch_df)
            batch_df["predicted_price"] = preds
            st.dataframe(batch_df)

            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions as CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Couldn't generate predictions — check your CSV's columns match what the model expects.\n\n{e}")

st.caption("Model: scikit-learn regression pipeline trained on the CarPrice dataset.")
