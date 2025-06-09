import streamlit as st
import base64
import io
import time
from utils.medical_rag import MedicalRAGProcessor
from utils.audio_processor import AudioProcessor
from config import MEDICAL_DISCLAIMER

# Set page configuration
st.set_page_config(
    page_title="Interactive Medical AI Coach", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .medical-disclaimer {
        background-color: #fef2f2;
        border: 2px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #dc2626;
        font-weight: 500;
    }
    
    .audio-section {
        background-color: #f0f9ff;
        border: 1px solid #0ea5e9;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .transcription-display {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .response-container {
        background-color: #f0fdf4;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .stAudio {
        margin: 1rem 0;
    }
    
    .processing-status {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = AudioProcessor()
    if 'rag_processor' not in st.session_state:
        st.session_state.rag_processor = MedicalRAGProcessor()
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None
    if 'processing_audio' not in st.session_state:
        st.session_state.processing_audio = False

def process_audio_transcription():
    """Process recorded audio and transcribe it"""
    if st.session_state.audio_file_path and not st.session_state.processing_audio:
        st.session_state.processing_audio = True
        
        with st.spinner("🎙️ Transcribing your audio..."):
            try:
                # Transcribe the audio
                transcription_result = st.session_state.audio_processor.transcribe_audio(
                    st.session_state.audio_file_path
                )
                
                if transcription_result['success']:
                    st.session_state.transcribed_text = transcription_result['text']
                    st.success("✅ Audio transcribed successfully!")
                    return True
                else:
                    st.error(f"❌ Transcription failed: {transcription_result['error']}")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Error during transcription: {str(e)}")
                return False
            finally:
                st.session_state.processing_audio = False
    return False

def process_medical_query(query_text, include_audio=True):
    """Process medical query and get response"""
    if not query_text.strip():
        st.warning("Please provide a medical question.")
        return None
    
    with st.spinner("🩺 Processing your medical query..."):
        try:
            # Call medical RAG processor
            response_data = st.session_state.rag_processor.process_medical_query(
                query=query_text,
                include_audio=include_audio
            )
            
            if response_data['success']:
                return response_data
            else:
                st.error(f"❌ Error processing query: {response_data.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"❌ Error during query processing: {str(e)}")
            return None

def main():
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Interactive Medical AI Coach</h1>
        <p>Powered by AWS Bedrock & RAG Architecture</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Medical disclaimer (prominent display)
    disclaimer_html = MEDICAL_DISCLAIMER.replace('•', '<br>•')
    st.markdown(f"""
    <div class="medical-disclaimer">
        <h3>⚠️ Important Medical Disclaimer</h3>
        {disclaimer_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Create two main sections
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎙️ Audio Input")
        st.markdown("""
        <div class="audio-section">
            <h4>Record Your Medical Question</h4>
            <p>Click record, ask your question, then click stop. The audio will be automatically transcribed.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio recording section
        if st.button("🎙️ Start Recording", key="start_record", type="primary"):
            if not st.session_state.is_recording:
                st.session_state.is_recording = True
                st.session_state.transcribed_text = ""
                with st.spinner("🎙️ Recording... (speak now)"):
                    time.sleep(0.5)  # Small delay for UI update
                    audio_file = st.session_state.audio_processor.record_audio(duration=10)
                    if audio_file:
                        st.session_state.audio_file_path = audio_file
                        st.success("✅ Recording completed!")
                        st.session_state.is_recording = False
                    else:
                        st.error("❌ Recording failed!")
                        st.session_state.is_recording = False
        
        # Show recording status
        if st.session_state.is_recording:
            st.markdown("""
            <div class="processing-status">
                🎙️ <strong>Recording in progress...</strong><br>
                Please speak clearly and wait for completion.
            </div>
            """, unsafe_allow_html=True)
        
        # Process transcription if audio exists
        if st.session_state.audio_file_path and not st.session_state.transcribed_text:
            if st.button("📝 Transcribe Audio", key="transcribe_btn"):
                process_audio_transcription()
        
        # Display audio player if audio exists
        if st.session_state.audio_file_path:
            st.audio(st.session_state.audio_file_path, format='audio/wav')
        
        # Display transcription
        if st.session_state.transcribed_text:
            st.markdown(f"""
            <div class="transcription-display">
                <h4>📝 Transcribed Text:</h4>
                <p><em>"{st.session_state.transcribed_text}"</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-process medical query button
            if st.button("🩺 Get Medical Response", key="process_transcription", type="primary"):
                response_data = process_medical_query(st.session_state.transcribed_text)
                if response_data:
                    st.session_state.current_response = response_data
    
    with col2:
        st.markdown("### ✍️ Text Input (Alternative)")
        
        # Text input as alternative
        manual_query = st.text_area(
            "Or type your medical question here:",
            placeholder="Example: What are the symptoms of diabetes? How can I manage high blood pressure?",
            height=100,
            key="manual_input"
        )
        
        if st.button("🩺 Process Text Query", key="process_text"):
            if manual_query.strip():
                response_data = process_medical_query(manual_query)
                if response_data:
                    st.session_state.current_response = response_data
            else:
                st.warning("Please enter a medical question.")
        
        # Show current input source
        if st.session_state.transcribed_text:
            st.info("🎙️ Audio input ready for processing")
        elif manual_query:
            st.info("✍️ Text input ready for processing")
    
    # Display response if available
    if hasattr(st.session_state, 'current_response') and st.session_state.current_response:
        st.markdown("---")
        response_data = st.session_state.current_response
        
        # Display response
        formatted_response = response_data['response'].replace('\n', '<br>')
        st.markdown(f"""
        <div class="response-container">
            <h3>🩺 Medical Information Response</h3>
            {formatted_response}
        </div>
        """, unsafe_allow_html=True)
        
        # Display confidence and context info
        col1, col2 = st.columns([1, 1])
        with col1:
            confidence = response_data.get('confidence', 0.0)
            st.metric("Confidence Level", f"{confidence:.1%}")
        
        with col2:
            context_count = len(response_data.get('contexts', []))
            st.metric("Knowledge Sources", context_count)
        
        # Audio response if available
        if response_data.get('audio_file'):
            st.markdown("### 🔊 Audio Response")
            st.audio(response_data['audio_file'], format='audio/mp3')
        
        # Show source contexts (expandable)
        if response_data.get('contexts'):
            with st.expander("📚 View Knowledge Base Sources"):
                for i, context in enumerate(response_data['contexts'], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(context[:300] + "..." if len(context) > 300 else context)
                    st.markdown("---")

# Sidebar with additional features
with st.sidebar:
    st.markdown("### 🛠️ System Status")
    
    # Connection status
    try:
        # Test AWS connection
        test_connection = st.session_state.rag_processor.test_aws_connection()
        if test_connection:
            st.success("✅ AWS Connection: Active")
        else:
            st.error("❌ AWS Connection: Failed")
    except:
        st.warning("⚠️ AWS Connection: Unknown")
    
    # Audio system status
    try:
        audio_status = st.session_state.audio_processor.test_audio_system()
        if audio_status:
            st.success("✅ Audio System: Ready")
        else:
            st.error("❌ Audio System: Failed")
    except:
        st.warning("⚠️ Audio System: Unknown")
    
    st.markdown("---")
    st.markdown("### 📋 Usage Instructions")
    st.markdown("""
    1. **Record Audio**: Click record and speak your question
    2. **Transcribe**: Audio is automatically converted to text
    3. **Process**: Get AI-powered medical information
    4. **Review**: Check confidence levels and sources
    
    **Tips:**
    - Speak clearly and avoid background noise
    - Ask specific medical questions
    - Review the medical disclaimer
    - Consult healthcare professionals for advice
    """)
    
    # Clear session button
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            if key not in ['audio_processor', 'rag_processor']:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()