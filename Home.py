import streamlit as st
import base64
import io
import time
import os
from dotenv import load_dotenv
from utils.medical_rag import MedicalRAGProcessor
from utils.audio_processor import AudioProcessor
from config import MEDICAL_DISCLAIMER

# CRITICAL: Load environment variables FIRST
load_dotenv()

# Verify environment variables are loaded
if not os.getenv('AWS_ACCESS_KEY_ID'):
    st.error("❌ AWS credentials not found. Please check your .env file.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Interactive Medical AI Coach", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like interface
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
    
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #dcfce7;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .ai-message {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .emergency-message {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #dc2626;
        font-weight: bold;
    }
    
    .input-container {
        background-color: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 20px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .recording-indicator {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .transcription-preview {
        background-color: #f1f5f9;
        border: 1px solid #64748b;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-style: italic;
    }
    
    .medical-disclaimer {
        background-color: #fef2f2;
        border: 2px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        margin: 2rem 0;
        color: #dc2626;
        font-size: 0.9rem;
    }
    
    .stButton > button {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = AudioProcessor()
    if 'rag_processor' not in st.session_state:
        st.session_state.rag_processor = MedicalRAGProcessor()
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def add_message_to_chat(role, content, message_type="normal"):
    """Add message to chat history"""
    st.session_state.chat_history.append({
        'role': role,
        'content': content,
        'type': message_type,
        'timestamp': time.time()
    })

def display_chat_history():
    """Display chat history in ChatGPT style"""
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="chat-container">
            <div class="ai-message">
                <strong>🩺 Medical AI Coach</strong><br>
                Hello! I'm here to help you with medical information. You can type your question or use the voice recording feature below.
                <br><br>
                <em>Remember: I provide general health information only. Always consult healthcare professionals for medical advice.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_html = '<div class="chat-container">'
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                chat_html += f'''
                <div class="user-message">
                    <strong>👤 You:</strong><br>
                    {message['content']}
                </div>
                '''
            elif message['role'] == 'assistant':
                css_class = "emergency-message" if message['type'] == 'emergency' else "ai-message"
                icon = "🚨" if message['type'] == 'emergency' else "🩺"
                
                chat_html += f'''
                <div class="{css_class}">
                    <strong>{icon} Medical AI Coach:</strong><br>
                    {message['content'].replace(chr(10), '<br>')}
                </div>
                '''
        
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

def process_audio_recording():
    """Handle audio recording and transcription"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.is_recording:
            if st.button("🎙️ Hold to Record", key="record_btn", type="secondary", use_container_width=True):
                st.session_state.is_recording = True
                st.rerun()
        else:
            st.markdown("""
            <div class="recording-indicator">
                🔴 <strong>Recording...</strong><br>
                <em>Speak clearly, release when done</em>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("⏹️ Stop Recording", key="stop_btn", type="primary", use_container_width=True):
                with st.spinner("🎙️ Processing your audio..."):
                    # Record audio
                    audio_file = st.session_state.audio_processor.record_audio(duration=10)
                    
                    if audio_file:
                        st.session_state.audio_file_path = audio_file
                        
                        # Transcribe immediately
                        transcription_result = st.session_state.audio_processor.transcribe_audio(audio_file)
                        
                        if transcription_result['success']:
                            st.session_state.transcribed_text = transcription_result['text']
                            st.success("✅ Audio transcribed successfully!")
                            
                            # Show transcription preview
                            st.markdown(f"""
                            <div class="transcription-preview">
                                <strong>Transcribed:</strong> "{st.session_state.transcribed_text}"
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.error(f"❌ Transcription failed: {transcription_result['error']}")
                    else:
                        st.error("❌ Recording failed!")
                
                st.session_state.is_recording = False
                st.rerun()

def process_user_query(query_text):
    """Process user query and add to chat"""
    if not query_text.strip():
        st.warning("Please provide a medical question.")
        return
    
    # Add user message to chat
    add_message_to_chat('user', query_text)
    
    # Process with AI
    with st.spinner("🩺 Analyzing your question..."):
        try:
            response_data = st.session_state.rag_processor.process_medical_query(
                query=query_text,
                include_audio=False  # Disable audio response for now to avoid TTS errors
            )
            
            if response_data['success']:
                message_type = 'emergency' if response_data.get('emergency', False) else 'normal'
                add_message_to_chat('assistant', response_data['response'], message_type)
                
                # Show additional info
                if response_data.get('contexts'):
                    with st.expander(f"📚 Sources ({len(response_data['contexts'])} found)"):
                        for i, context in enumerate(response_data['contexts'], 1):
                            st.text(f"Source {i}: {context[:200]}...")
            else:
                error_response = f"I apologize, but I'm having technical difficulties. Error: {response_data.get('error', 'Unknown error')}"
                add_message_to_chat('assistant', error_response)
                
        except Exception as e:
            error_response = f"I'm currently experiencing technical issues. Please try again later. Error: {str(e)}"
            add_message_to_chat('assistant', error_response)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Interactive Medical AI Coach</h1>
        <p>Ask questions using voice or text • Powered by AWS Bedrock & RAG Architecture</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat display
    display_chat_history()
    
    # Input section (ChatGPT style)
    st.markdown("""
    <div class="input-container">
    """, unsafe_allow_html=True)
    
    # Text input with voice option
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Show transcribed text if available
        if st.session_state.transcribed_text:
            user_input = st.text_area(
                "Your medical question:", 
                value=st.session_state.transcribed_text,
                placeholder="Type your medical question here or use voice recording...",
                height=80,
                key="text_input"
            )
        else:
            user_input = st.text_area(
                "Your medical question:", 
                placeholder="Type your medical question here or use voice recording...",
                height=80,
                key="text_input"
            )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("🎙️", help="Voice Input", type="secondary"):
            st.session_state.show_voice_input = not st.session_state.get('show_voice_input', False)
            st.rerun()
    
    # Voice input section (toggleable)
    if st.session_state.get('show_voice_input', False):
        st.markdown("### 🎙️ Voice Input")
        process_audio_recording()
        
        # Clear transcribed text button
        if st.session_state.transcribed_text:
            if st.button("🗑️ Clear Transcription"):
                st.session_state.transcribed_text = ""
                st.rerun()
    
    # Send button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Send 📤", type="primary", use_container_width=True):
            if user_input.strip():
                process_user_query(user_input.strip())
                # Clear input
                st.session_state.transcribed_text = ""
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.transcribed_text = ""
            st.rerun()
    
    # Medical disclaimer at bottom
    st.markdown("---")
    disclaimer_html = MEDICAL_DISCLAIMER.replace('•', '<br>•')
    st.markdown(f"""
    <div class="medical-disclaimer">
        <h3>⚠️ Important Medical Disclaimer</h3>
        {disclaimer_html}
    """, unsafe_allow_html=True)

# Sidebar with system status
with st.sidebar:
    st.markdown("### 🛠️ System Status")
    
    # Environment variables check
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    if aws_access_key:
        st.success(f"✅ AWS Credentials: Loaded")
        st.text(f"Key: {aws_access_key[:10]}...")
    else:
        st.error("❌ AWS Credentials: Missing")
    
    # AWS connection test
    try:
        if hasattr(st.session_state, 'rag_processor'):
            test_connection = st.session_state.rag_processor.test_aws_connection()
            if test_connection:
                st.success("✅ AWS Connection: Active")
            else:
                st.error("❌ AWS Connection: Failed")
        else:
            st.warning("⚠️ AWS Connection: Initializing...")
    except Exception as e:
        st.error("❌ AWS Connection: Error")
        st.text(f"Error: {str(e)[:50]}...")
    
    # Audio system status
    try:
        if hasattr(st.session_state, 'audio_processor'):
            audio_status = st.session_state.audio_processor.test_audio_system()
            if audio_status:
                st.success("✅ Audio System: Ready")
            else:
                st.error("❌ Audio System: Failed")
        else:
            st.warning("⚠️ Audio System: Initializing...")
    except Exception as e:
        st.error("❌ Audio System: Error")
    
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    **💬 Chat Interface:**
    - Type questions in the text area
    - Click 🎙️ for voice input
    - Press Send 📤 to get responses
    
    **🎙️ Voice Input:**
    - Click the microphone button
    - Hold "Record" and speak clearly
    - Release to transcribe automatically
    
    **💡 Tips:**
    - Ask specific medical questions
    - Review sources in expandable sections
    - Emergency symptoms trigger safety alerts
    """)
    
    # Debug information
    with st.expander("🔧 Debug Info"):
        st.text(f"Chat messages: {len(st.session_state.get('chat_history', []))}")
        st.text(f"Recording: {st.session_state.get('is_recording', False)}")
        st.text(f"Transcribed: {bool(st.session_state.get('transcribed_text', ''))}")
        st.text(f"Environment loaded: {bool(os.getenv('AWS_ACCESS_KEY_ID'))}")

if __name__ == "__main__":
    main()