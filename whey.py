import joblib
import pandas as pd
import streamlit as st
import base64

st.set_page_config(
    page_title="WheyGuard",
    layout="centered",
    initial_sidebar_state="auto"
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("D:\wheyguard\Airbrush-image-extender (2).jpeg")


model = joblib.load('D:\wheyguard\whey_model.joblib')
encoder = joblib.load('D:\wheyguard\onehotencoder.joblib')
feature = joblib.load('D:\wheyguard\data.joblib')
scaler = joblib.load('D:\wheyguard\scaler.joblib')

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>WheyGuard</h1>", unsafe_allow_html=True)
st.subheader("Whey Protein Quality Checker")
st.info("""Welcome to **WheyGuard** — an app to evaluate whey protein quality.  
    Enter the sample measurements in the left side bar and press **Evaluate** to get an estimated quality of your whey protein.""")

st.sidebar.header("Enter the Values")
with st.sidebar.form(key='input_form'):
    product_type = st.selectbox("Product type", options=["isolate", "concentrate", "blend"], index=0)
    protein_pct = st.slider("Protein (%)", 0.0, 90.0, 85.0, step=0.1)
    bcaa_g = st.slider("BCAA (g per 100g)", 0.0, 20.0, 12.0, step=0.01)
    leucine_g = st.slider("Leucine (g per 100g)", 0.0, 15.0, 4.5, step=0.01)
    denaturation_pct = st.slider("Denaturation (%)", 0.0, 50.0, 6.0, step=0.1)
    odor_score = st.slider("Odor (1-10)", 0.0, 10.0, 8.0, step=0.1)
    flavor_score = st.slider("Flavor (1-10)", 0.0, 10.0, 8.0, step=0.1)
    cadmium_ppm = st.number_input("Cadmium (ppm)", min_value=0.0, max_value=0.0335, value=0.001, format="%f")
    ftir_pc1 = st.number_input("FTIR_PC1 (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=4.0, format="%f")
    submitted = st.form_submit_button(label=' Evaluate')

input_data = [[product_type, protein_pct, bcaa_g, leucine_g, denaturation_pct,odor_score, flavor_score, cadmium_ppm, ftir_pc1]]
input_df = pd.DataFrame(input_data, columns=feature.columns)
input_encoded = encoder.transform(input_df)
input_scaled = scaler.transform(input_encoded)
pred = model.predict(input_scaled)

if submitted:
    if pred[0] == "High":
        st.success(" Predicted Quality: HIGH")
        st.metric(label="Protein Purity", value="Excellent")
    elif pred[0] == "Medium":
        st.warning("Predicted Quality: MEDIUM")
        st.metric(label="Protein Purity", value="Moderate")
    else:
        st.error("Predicted Quality: LOW")
        st.metric(label="Protein Purity", value="Poor")

st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: grey;'>
        Built By  <a href="https://www.instagram.com/gymbruu?igsh=MXIzM2trbGt1dWwweg==" target="_blank" style="color:#0A66C2; text-decoration:none;"><b>gymbruu</b></a>
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style='text-align: center; color: grey;'>
        Check Out My  <a href="https://www.linkedin.com/in/mohammed-irfan-7b1548323" target="_blank" style="color:#0A66C2; text-decoration:none;"><b>LinkedIn</b></a>
    </p>
    """,
    unsafe_allow_html=True
)
