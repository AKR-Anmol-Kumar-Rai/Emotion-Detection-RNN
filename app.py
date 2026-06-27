import streamlit as st
import torch
import joblib
from model import RNN
from utils import preprocess, label_map

# page config
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🧠",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
/* Main background */
.stApp {
    background: radial-gradient(circle at top left, #2e1065, #020617 45%, #000814);
    color: white;
}

/* Hide default streamlit styling */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Title */
.title {
    text-align: center;
    font-size: 70px;
    font-weight: 800;
    background: linear-gradient(to right, #f472b6, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 40px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
    margin-bottom: 50px;
}

/* Input */
.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.03);
    color: white;
    border: 2px solid #9333ea;
    border-radius: 18px;
    padding: 18px;
    font-size: 20px;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #9333ea);
    color: white;
    font-size: 22px;
    font-weight: bold;
    border-radius: 15px;
    padding: 15px 35px;
    border: none;
    box-shadow: 0px 0px 20px rgba(147,51,234,0.7);
    transition: 0.3s ease-in-out;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 30px rgba(59,130,246,0.9);
}

/* Result card */
.result-card {
    margin-top: 50px;
    padding: 40px;
    border-radius: 25px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    text-align: center;
}

/* Emoji */
.emoji {
    font-size: 80px;
}

/* Emotion text */
.emotion-text {
    font-size: 90px;
    font-weight: 900;
    color: #22c55e;
    text-shadow: 0px 0px 20px rgba(34,197,94,0.8);
}

/* Footer */
.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 40px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Load model
device = torch.device("cpu")

tf = joblib.load("tfidf.pkl")

model = RNN(input_size=15154)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# UI
st.markdown('<div class="title">🧠 Emotion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze emotions from text using RNN</div>', unsafe_allow_html=True)

text = st.text_input("Enter your sentence")

emoji_map = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲"
}

if st.button("✨ Predict Emotion"):
    clean_text = preprocess(text)

    vector = tf.transform([clean_text])
    vector = torch.tensor(vector.toarray(), dtype=torch.float32)

    vector = vector.unsqueeze(1)

    with torch.no_grad():
        output = model(vector)
        pred = torch.argmax(output, dim=1).item()

    emotion = label_map[pred]
    emoji = emoji_map[emotion]

    st.markdown(f"""
    <div class="result-card">
        <div class="emoji">{emoji}</div>
        <div class="emotion-text">{emotion.upper()}</div>
        <br>
        <p style="font-size:22px;color:#94a3b8;">Detected Emotion</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">💜 Built with Streamlit & PyTorch</div>', unsafe_allow_html=True)