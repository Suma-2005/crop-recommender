import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Crop Recommender", page_icon="🌾", layout="centered")

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #2e7d32, #66bb6a);
    padding: 2rem; border-radius: 16px; text-align: center;
    color: white; margin-bottom: 1.5rem;
}
.hero h1 { color: white; margin: 0; }
.hero p { margin: 0.5rem 0 0 0; font-size: 1.05rem; color: white; }
.result {
    background: #e8f5e9; border: 2px solid #66bb6a; border-radius: 16px;
    padding: 1.5rem; text-align: center; margin-top: 1rem;
}
.result .emoji { font-size: 4rem; }
.result h2 { color: #1b5e20; margin: 0.3rem 0; }
.result p { color: #2e7d32; margin: 0; }
</style>
""", unsafe_allow_html=True)

EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫛", "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "cotton": "☁️",
    "jute": "🌿", "coffee": "☕",
}

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.markdown("""
<div class="hero">
    <h1>🌾 Crop Recommendation System</h1>
    <p>Enter your soil and weather details to find the best crop to grow.</p>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("🧪 Soil details")
    c1, c2 = st.columns(2)
    N = c1.slider("Nitrogen (N)", 0, 140, 50)
    P = c2.slider("Phosphorus (P)", 5, 145, 50)
    K = c1.slider("Potassium (K)", 5, 205, 50)
    ph = c2.slider("Soil pH", 3.5, 10.0, 6.5, 0.1)

with st.container(border=True):
    st.subheader("🌦️ Weather details")
    c3, c4, c5 = st.columns(3)
    temp = c3.slider("Temperature (°C)", 8.0, 44.0, 25.0)
    humidity = c4.slider("Humidity (%)", 14.0, 100.0, 70.0)
    rainfall = c5.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

if st.button("🔍 Recommend Crop", use_container_width=True, type="primary"):
    data = pd.DataFrame(
        [[N, P, K, temp, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )
    probs = model.predict_proba(data)[0]
    top3 = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
    best_crop, best_prob = top3[0]

    st.markdown(f"""
    <div class="result">
        <div class="emoji">{EMOJI.get(best_crop, "🌱")}</div>
        <p>Best crop for your land</p>
        <h2>{best_crop.title()}</h2>
        <p>Confidence: {best_prob * 100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Other good options")
    for crop, p in top3[1:]:
        st.write(f"{EMOJI.get(crop, '🌱')} **{crop.title()}**")
        st.progress(float(p), text=f"{p * 100:.1f}%")

st.caption("Built with Python, scikit-learn and Streamlit.")