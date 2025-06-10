import streamlit as st
import base64
import io
import time
import os
from dotenv import load_dotenv
from utils.medical_rag import MedicalRAGProcessor
from utils.audio_processor import AudioProcessor
from config import MEDICAL_DISCLAIMER

# Initialize logging FIRST
from utils.logging import setup_logging, log_function_call, log_user_interaction
logger = setup_logging()

# CRITICAL: Load environment variables FIRST
logger.info("🔧 Loading environment variables...")
load_dotenv()

# Verify environment variables are loaded
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
if not aws_key:
    logger.error("❌ AWS credentials not found in environment variables")
    st.error("❌ AWS credentials not found. Please check your .env file.")
    st.stop()
else:
    logger.info(f"✅ AWS credentials loaded - Key: {aws_key[:10]}...")

# Set page configuration
logger.debug("🎨 Setting Streamlit page configuration...")
st.set_page_config(
    page_title="Interactive Medical AI Coach", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like interface
logger.debug("🎨 Loading custom CSS...")
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

@log_function_call
def initialize_session_state():
    """Initialize session state variables with logging"""
    logger.debug("🔧 Initializing Streamlit session state...")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        logger.debug("✅ Initialized chat_history")
    
    if 'audio_processor' not in st.session_state:
        logger.debug("🎙️ Initializing AudioProcessor...")
        st.session_state.audio_processor = AudioProcessor()
        logger.debug("✅ AudioProcessor initialized")
    
    if 'rag_processor' not in st.session_state:
        logger.debug("🧠 Initializing MedicalRAGProcessor...")
        st.session_state.rag_processor = MedicalRAGProcessor()
        logger.debug("✅ MedicalRAGProcessor initialized")
    
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
        logger.debug("✅ Initialized is_recording")
    
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
        logger.debug("✅ Initialized transcribed_text")
    
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None
        logger.debug("✅ Initialized audio_file_path")
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
        logger.debug("✅ Initialized processing")
    
    logger.info("🎯 Session state initialization completed")

@log_function_call
def add_message_to_chat(role, content, message_type="normal"):
    """Add message to chat history with logging"""
    logger.bind(category="user").info(f"💬 Adding message - Role: {role}, Type: {message_type}")
    logger.debug(f"Message content preview: {content[:100]}...")
    
    message = {
        'role': role,
        'content': content,
        'type': message_type,
        'timestamp': time.time()
    }
    
    st.session_state.chat_history.append(message)
    logger.debug(f"✅ Message added. Total messages: {len(st.session_state.chat_history)}")

@log_function_call
def display_chat_history():
    """Display chat history in ChatGPT style with logging"""
    logger.debug(f"🖥️ Displaying chat history - {len(st.session_state.chat_history)} messages")
    
    if not st.session_state.chat_history:
        logger.debug("📝 Displaying welcome message")
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
        logger.debug(f"📝 Rendering {len(st.session_state.chat_history)} chat messages")
        chat_html = '<div class="chat-container">'
        
        for i, message in enumerate(st.session_state.chat_history):
            logger.debug(f"Rendering message {i+1}: {message['role']} - {message['type']}")
            
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
        logger.debug("✅ Chat history rendered successfully")

@log_function_call
@log_user_interaction("Audio Recording")
def process_audio_recording():
    """Handle audio recording and transcription with logging"""
    logger.debug("🎙️ Processing audio recording interface")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.is_recording:
            if st.button("🎙️ Hold to Record", key="record_btn", type="secondary", use_container_width=True):
                logger.bind(category="user").info("🎙️ User started recording")
                st.session_state.is_recording = True
                st.rerun()
        else:
            logger.debug("🔴 Currently recording - showing recording interface")
            st.markdown("""
            <div class="recording-indicator">
                🔴 <strong>Recording...</strong><br>
                <em>Speak clearly, release when done</em>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("⏹️ Stop Recording", key="stop_btn", type="primary", use_container_width=True):
                logger.bind(category="user").info("⏹️ User stopped recording")
                
                with st.spinner("🎙️ Processing your audio..."):
                    logger.debug("🎙️ Starting audio processing...")
                    
                    # Record audio
                    audio_file = st.session_state.audio_processor.record_audio(duration=10)
                    
                    if audio_file:
                        logger.info(f"✅ Audio recorded successfully: {audio_file}")
                        st.session_state.audio_file_path = audio_file
                        
                        # Transcribe immediately
                        logger.debug("🔤 Starting audio transcription...")
                        transcription_result = st.session_state.audio_processor.transcribe_audio(audio_file)
                        
                        if transcription_result['success']:
                            logger.info(f"✅ Audio transcribed: '{transcription_result['text'][:50]}...'")
                            st.session_state.transcribed_text = transcription_result['text']
                            st.success("✅ Audio transcribed successfully!")
                            
                            # Show transcription preview
                            st.markdown(f"""
                            <div class="transcription-preview">
                                <strong>Transcribed:</strong> "{st.session_state.transcribed_text}"
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            logger.error(f"❌ Transcription failed: {transcription_result['error']}")
                            st.error(f"❌ Transcription failed: {transcription_result['error']}")
                    else:
                        logger.error("❌ Audio recording failed")
                        st.error("❌ Recording failed!")
                
                st.session_state.is_recording = False
                st.rerun()

@log_function_call
@log_user_interaction("Medical Query Processing")
def process_user_query(query_text):
    """Process user query and add to chat with comprehensive logging"""
    logger.bind(category="user").info(f"🔍 Processing user query: '{query_text[:100]}...'")
    
    if not query_text.strip():
        logger.warning("⚠️ Empty query provided")
        st.warning("Please provide a medical question.")
        return
    
    # Add user message to chat
    logger.debug("💬 Adding user query to chat history")
    add_message_to_chat('user', query_text)
    
    # Process with AI
    with st.spinner("🩺 Analyzing your question..."):
        logger.debug("🧠 Starting AI analysis...")
        
        try:
            logger.debug("📡 Calling MedicalRAGProcessor.process_medical_query()")
            response_data = st.session_state.rag_processor.process_medical_query(
                query=query_text,
                include_audio=False  # Disable audio response for now to avoid TTS errors
            )
            
            if response_data['success']:
                logger.info("✅ AI processing successful")
                logger.debug(f"Response confidence: {response_data.get('confidence', 'unknown')}")
                logger.debug(f"Emergency detected: {response_data.get('emergency', False)}")
                logger.debug(f"Contexts found: {len(response_data.get('contexts', []))}")
                
                message_type = 'emergency' if response_data.get('emergency', False) else 'normal'
                add_message_to_chat('assistant', response_data['response'], message_type)
                
                # Show additional info
                if response_data.get('contexts'):
                    logger.debug(f"📚 Displaying {len(response_data['contexts'])} knowledge base sources")
                    with st.expander(f"📚 Sources ({len(response_data['contexts'])} found)"):
                        for i, context in enumerate(response_data['contexts'], 1):
                            st.text(f"Source {i}: {context[:200]}...")
            else:
                logger.error(f"❌ AI processing failed: {response_data.get('error', 'Unknown error')}")
                error_response = f"I apologize, but I'm having technical difficulties. Error: {response_data.get('error', 'Unknown error')}"
                add_message_to_chat('assistant', error_response)
                
        except Exception as e:
            logger.error(f"💥 Exception during query processing: {str(e)}")
            logger.exception("Full exception details:")
            error_response = f"I'm currently experiencing technical issues. Please try again later. Error: {str(e)}"
            add_message_to_chat('assistant', error_response)

@log_function_call
def main():
    """Main application function with comprehensive logging"""
    logger.info("🚀 Starting Medical AI Coach application")
    
    # Initialize session state
    initialize_session_state()
    
    # Main header
    logger.debug("🎨 Rendering main header")
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Interactive Medical AI Coach</h1>
        <p>Ask questions using voice or text • Powered by AWS Bedrock</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat display
    logger.debug("💬 Displaying chat interface")
    display_chat_history()
    
    # Input section (ChatGPT style)
    logger.debug("📝 Rendering input interface")
    st.markdown("""
    <div class="input-container">
    """, unsafe_allow_html=True)
    
    # Text input with voice option
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Show transcribed text if available
        if st.session_state.transcribed_text:
            logger.debug(f"📝 Showing transcribed text: '{st.session_state.transcribed_text[:50]}...'")
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
            logger.bind(category="user").info("🎙️ User clicked voice input toggle")
            st.session_state.show_voice_input = not st.session_state.get('show_voice_input', False)
            st.rerun()
    
    # Voice input section (toggleable)
    if st.session_state.get('show_voice_input', False):
        logger.debug("🎙️ Displaying voice input interface")
        st.markdown("### 🎙️ Voice Input")
        process_audio_recording()
        
        # Clear transcribed text button
        if st.session_state.transcribed_text:
            if st.button("🗑️ Clear Transcription"):
                logger.bind(category="user").info("🗑️ User cleared transcription")
                st.session_state.transcribed_text = ""
                st.rerun()
    
    # Send button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Send 📤", type="primary", use_container_width=True):
            if user_input.strip():
                logger.bind(category="user").info("📤 User clicked Send button")
                process_user_query(user_input.strip())
                # Clear input
                st.session_state.transcribed_text = ""
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            logger.bind(category="user").info(f"🗑️ User cleared chat history ({len(st.session_state.chat_history)} messages)")
            st.session_state.chat_history = []
            st.session_state.transcribed_text = ""
            st.rerun()
    
    # Medical disclaimer at bottom
    logger.debug("⚠️ Displaying medical disclaimer")
    st.markdown("---")
    disclaimer_html = MEDICAL_DISCLAIMER.replace('•', '<br>•')
    st.markdown(f"""
    <div class="medical-disclaimer">
        <h3>⚠️ Important Medical Disclaimer</h3>
        {disclaimer_html}
    </div>
    """, unsafe_allow_html=True)

# Sidebar with system status
logger.debug("📊 Creating sidebar system status")
with st.sidebar:
    st.markdown("### 🛠️ System Status")
    
    # Environment variables check
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    if aws_access_key:
        logger.debug(f"✅ AWS credentials found in sidebar check: {aws_access_key[:10]}...")
        st.success(f"✅ AWS Credentials: Loaded")
        st.text(f"Key: {aws_access_key[:10]}...")
    else:
        logger.error("❌ AWS credentials missing in sidebar check")
        st.error("❌ AWS Credentials: Missing")
    
    # AWS connection test
    try:
        if hasattr(st.session_state, 'rag_processor'):
            logger.debug("🧪 Testing AWS connection...")
            test_connection = st.session_state.rag_processor.test_aws_connection()
            if test_connection:
                logger.info("✅ AWS connection test successful")
                st.success("✅ AWS Connection: Active")
            else:
                logger.error("❌ AWS connection test failed")
                st.error("❌ AWS Connection: Failed")
        else:
            logger.warning("⚠️ RAG processor not initialized yet")
            st.warning("⚠️ AWS Connection: Initializing...")
    except Exception as e:
        logger.error(f"💥 AWS connection test exception: {str(e)}")
        st.error("❌ AWS Connection: Error")
        st.text(f"Error: {str(e)[:50]}...")
    
    # Audio system status
    try:
        if hasattr(st.session_state, 'audio_processor'):
            logger.debug("🧪 Testing audio system...")
            audio_status = st.session_state.audio_processor.test_audio_system()
            if audio_status:
                logger.info("✅ Audio system test successful")
                st.success("✅ Audio System: Ready")
            else:
                logger.error("❌ Audio system test failed")
                st.error("❌ Audio System: Failed")
        else:
            logger.warning("⚠️ Audio processor not initialized yet")
            st.warning("⚠️ Audio System: Initializing...")
    except Exception as e:
        logger.error(f"💥 Audio system test exception: {str(e)}")
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
        
        # Logging status
        st.text(f"Log files created: {len([f for f in os.listdir('logs') if f.endswith('.log')])}" if os.path.exists('logs') else "Logs folder: Not created")

if __name__ == "__main__":
    logger.info("🎬 Application entry point reached")
    main()
    logger.info("🎬 Application main() completed")