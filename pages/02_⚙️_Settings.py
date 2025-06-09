import streamlit as st
from config import MEDICAL_DISCLAIMER

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ Medical AI Coach Settings")

# Model settings
st.subheader("🤖 AI Model Configuration")

col1, col2 = st.columns(2)

with col1:
    model_temperature = st.slider("Response Creativity", 0.0, 1.0, 0.3, 0.1,
                                 help="Lower values = more conservative responses")
    
    max_tokens = st.slider("Max Response Length", 100, 2000, 1000, 100)

with col2:
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05,
                                   help="Minimum confidence for providing responses")
    
    num_sources = st.slider("Knowledge Sources to Use", 1, 10, 5)

# Audio settings
st.subheader("🔊 Audio Configuration")

col1, col2 = st.columns(2)

with col1:
    voice_selection = st.selectbox("Voice Selection", 
                                  ["Joanna (Female)", "Matthew (Male)", "Amy (Female)", "Brian (Male)"])
    
    speech_rate = st.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1)

with col2:
    audio_quality = st.selectbox("Audio Quality", ["Standard", "High", "Neural"])
    
    auto_play = st.checkbox("Auto-play audio responses", value=True)

# Safety settings
st.subheader("🛡️ Safety Configuration")

emergency_detection = st.checkbox("Enable Emergency Keyword Detection", value=True)
strict_disclaimers = st.checkbox("Show Detailed Medical Disclaimers", value=True)
professional_referral = st.checkbox("Always Recommend Professional Consultation", value=True)

# Save settings
if st.button("💾 Save Settings"):
    st.success("Settings saved successfully!")

# Medical disclaimer
st.markdown("---")
st.subheader("⚠️ Medical Disclaimer")
st.text_area("Current Medical Disclaimer:", value=MEDICAL_DISCLAIMER, height=200, disabled=True)