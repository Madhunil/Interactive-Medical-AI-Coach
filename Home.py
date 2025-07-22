import streamlit as st
import base64
import io
import time
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.medical_rag import MedicalRAGProcessor
from utils.audio_processor import AudioProcessor
from utils.session_analytics import SessionAnalytics
from config import MEDICAL_DISCLAIMER, THERAPEUTIC_AREAS

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

# Set page configuration with enhanced styling
st.set_page_config(
    page_title="Interactive Medical AI Coach", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified CSS for better compatibility
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .status-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .status-success {
        border-left: 4px solid #22c55e;
        background: #f0fdf4;
    }
    
    .status-warning {
        border-left: 4px solid #f59e0b;
        background: #fffbeb;
    }
    
    .status-error {
        border-left: 4px solid #dc2626;
        background: #fef2f2;
    }
    
    .therapeutic-area-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .therapeutic-area-card:hover {
        border-color: #3b82f6;
        background: #f0f9ff;
    }
    
    .emergency-alert {
        background: #fef2f2;
        border: 2px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        color: #dc2626;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
</style>
""", unsafe_allow_html=True)

@log_function_call
def initialize_session_state():
    """Initialize enhanced session state with Lambda configuration"""
    logger.debug("🔧 Initializing Streamlit session state...")
    
    # Core components with robust error handling
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'audio_processor' not in st.session_state:
        try:
            st.session_state.audio_processor = AudioProcessor()
            logger.info("✅ AudioProcessor initialized successfully")
        except Exception as e:
            logger.error(f"❌ AudioProcessor initialization failed: {str(e)}")
            st.session_state.audio_processor = None
    
    if 'rag_processor' not in st.session_state:
        try:
            st.session_state.rag_processor = MedicalRAGProcessor()
            logger.info("✅ RAG processor initialized successfully")
        except Exception as e:
            logger.error(f"❌ RAG processor initialization failed: {str(e)}")
            st.error(f"❌ Failed to initialize medical AI processor: {str(e)}")
            st.stop()
    
    if 'session_analytics' not in st.session_state:
        try:
            st.session_state.session_analytics = SessionAnalytics()
            logger.info("✅ Session analytics initialized successfully")
        except Exception as e:
            logger.error(f"❌ Session analytics initialization failed: {str(e)}")
            st.session_state.session_analytics = None
    
    # Session management
    if 'selected_therapeutic_area' not in st.session_state:
        st.session_state.selected_therapeutic_area = None
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    
    # Audio and recording states
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    
    if 'show_voice_input' not in st.session_state:
        st.session_state.show_voice_input = False
    
    # Lambda integration settings
    if 'use_lambda_integration' not in st.session_state:
        st.session_state.use_lambda_integration = True  # Enable by default
    
    if 'lambda_status' not in st.session_state:
        st.session_state.lambda_status = None
    
    logger.info("🎯 Session state initialization completed")

@log_function_call
def test_lambda_connection():
    """Test Lambda function connectivity"""
    if not st.session_state.rag_processor:
        return False, "RAG processor not initialized"
    
    try:
        # Test basic Lambda connectivity
        test_payload = {
            'api_Path': 'getStory',
            'story_theme': 'medical test query',
            'story_type': 'medical',
            'main_character': 'Doctor',
            'story_lang': 'English',
            'word_count': '100'
        }
        
        result = st.session_state.rag_processor.call_lambda_function(test_payload)
        
        if result['success']:
            return True, "Lambda function accessible"
        else:
            return False, f"Lambda error: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return False, f"Lambda test failed: {str(e)}"

@log_function_call
def render_lambda_status():
    """Render Lambda integration status"""
    # Test Lambda connection
    lambda_working, lambda_message = test_lambda_connection()
    st.session_state.lambda_status = lambda_working
    
    if lambda_working:
        status_class = "status-success"
        status_icon = "✅"
        status_text = "Lambda Integration Active"
    else:
        status_class = "status-warning"
        status_icon = "⚠️"
        status_text = "Lambda Integration Limited"
    
    st.markdown(f"""
    <div class="status-card {status_class}">
        <h4>{status_icon} {status_text}</h4>
        <p><strong>Function:</strong> wonderscribeconnectVDB</p>
        <p><strong>Status:</strong> {lambda_message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Lambda settings toggle
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.session_state.use_lambda_integration = st.checkbox(
            "🔧 Use Lambda Integration for Enhanced Processing",
            value=st.session_state.use_lambda_integration,
            help="Enable to use Lambda function for enhanced medical query processing"
        )
    
    with col2:
        if st.button("🔄 Test Lambda"):
            st.rerun()

@log_function_call
def render_therapeutic_area_selector():
    """Render therapeutic area selection interface"""
    if st.session_state.selected_therapeutic_area:
        # Show current selection with option to change
        area_info = THERAPEUTIC_AREAS[st.session_state.selected_therapeutic_area]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"{area_info['icon']} **Current Area:** {area_info['name']} - {area_info['description']}")
        
        with col2:
            if st.button("🔄 Change Area", type="secondary"):
                st.session_state.selected_therapeutic_area = None
                st.rerun()
        
        return True
    
    else:
        st.markdown("### 🎯 Select Your Therapeutic Area of Interest")
        st.markdown("Choose a medical specialty to get specialized knowledge and guidance:")
        
        # Create area selection grid
        cols = st.columns(3)
        
        for i, (area_key, area_info) in enumerate(THERAPEUTIC_AREAS.items()):
            with cols[i % 3]:
                if st.button(
                    f"{area_info['icon']}\n{area_info['name']}", 
                    key=f"area_{area_key}",
                    help=area_info['description'],
                    use_container_width=True
                ):
                    st.session_state.selected_therapeutic_area = area_key
                    st.session_state.rag_processor.set_therapeutic_area(area_key)
                    st.success(f"✅ Selected {area_info['name']}!")
                    st.rerun()
        
        return False

@log_function_call
def render_session_dashboard():
    """Render session progress dashboard with Lambda status"""
    st.markdown("### 📊 Current Session")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate session stats
    session_duration = (datetime.now() - st.session_state.session_start_time).total_seconds() / 60
    questions_asked = len([msg for msg in st.session_state.chat_history if msg['role'] == 'user'])
    
    with col1:
        st.metric("Session Duration", f"{session_duration:.0f} min")
    
    with col2:
        st.metric("Questions Asked", questions_asked)
    
    with col3:
        if st.session_state.selected_therapeutic_area:
            area_name = THERAPEUTIC_AREAS[st.session_state.selected_therapeutic_area]['name']
            st.metric("Current Area", area_name)
        else:
            st.metric("Current Area", "None Selected")
    
    with col4:
        if st.session_state.chat_history:
            confidence_score = st.session_state.session_analytics.calculate_session_confidence(
                st.session_state.chat_history
            )
            st.metric("Avg Confidence", f"{confidence_score:.0f}%")
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col5:
        lambda_status = "Active" if st.session_state.lambda_status else "Limited"
        st.metric("Lambda Status", lambda_status)

@log_function_call
def add_message_to_chat(role, content, message_type="normal", metadata=None):
    """Enhanced message addition with metadata and Lambda tracking"""
    message = {
        'role': role,
        'content': content,
        'type': message_type,
        'timestamp': time.time(),
        'therapeutic_area': st.session_state.selected_therapeutic_area,
        'metadata': metadata or {},
        'lambda_enhanced': metadata and 'lambda_used' in metadata and metadata['lambda_used']
    }
    
    st.session_state.chat_history.append(message)
    
    # Log analytics
    if st.session_state.session_analytics:
        st.session_state.session_analytics.log_interaction(
            st.session_state.session_id,
            role,
            content,
            st.session_state.selected_therapeutic_area
        )

@log_function_call
def display_enhanced_chat_history():
    """Display enhanced chat history using Streamlit native components"""
    if not st.session_state.chat_history:
        welcome_area = st.session_state.selected_therapeutic_area
        area_info = THERAPEUTIC_AREAS.get(welcome_area, {'name': 'General Medicine', 'icon': '🩺'})
        
        # Welcome message
        st.info(f"""
        **{area_info['icon']} Medical AI Coach - {area_info['name']}**
        
        Welcome to the Interactive Medical AI Coach! I'm here to help you learn about {area_info['name'].lower()}. 
        You can ask questions using text or voice input.
        
        💡 **Tip:** Try asking about specific conditions, treatments, drug mechanisms, or clinical guidelines.
        
        ⚠️ **Remember:** This is for educational purposes only. Always follow clinical guidelines and consult healthcare professionals.
        """)
        
        # Lambda integration info
        if st.session_state.use_lambda_integration and st.session_state.lambda_status:
            st.success("🔧 **Lambda Integration:** Enhanced processing with wonderscribeconnectVDB function enabled")
        
        # Show quick action buttons for the selected area
        if welcome_area and welcome_area in THERAPEUTIC_AREAS:
            area_info = THERAPEUTIC_AREAS[welcome_area]
            if 'common_questions' in area_info:
                st.markdown("#### 🚀 Quick Start Questions:")
                quick_cols = st.columns(2)
                
                for i, question in enumerate(area_info['common_questions'][:4]):
                    with quick_cols[i % 2]:
                        if st.button(f"💭 {question}", key=f"quick_{i}", use_container_width=True):
                            st.session_state.transcribed_text = question
                            st.rerun()
    else:
        # Display chat messages using Streamlit's chat interface
        st.markdown("### 💬 Conversation History")
        
        # Create container for chat messages
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                timestamp = datetime.fromtimestamp(message['timestamp']).strftime("%H:%M")
                
                if message['role'] == 'user':
                    # User message
                    with st.chat_message("user"):
                        st.write(f"**You** • {timestamp}")
                        st.write(message['content'])
                
                elif message['role'] == 'assistant':
                    # AI message with indicators
                    avatar = "🚨" if message['type'] == 'emergency' else ("🔧" if message.get('lambda_enhanced') else "🧠")
                    
                    with st.chat_message("assistant", avatar=avatar):
                        # Header with indicators
                        indicators = []
                        if 'metadata' in message and 'confidence' in message['metadata']:
                            confidence = message['metadata']['confidence']
                            if confidence < 0.7:
                                indicators.append(f"⚠️ {confidence:.0%} confidence")
                            else:
                                indicators.append(f"✅ {confidence:.0%} confidence")
                        
                        if message.get('lambda_enhanced'):
                            indicators.append("🔧 Lambda Enhanced")
                        
                        if message['type'] == 'emergency':
                            indicators.append("🚨 Emergency Alert")
                        
                        indicator_text = " • ".join(indicators)
                        header_text = f"**AI Coach** • {timestamp}"
                        if indicator_text:
                            header_text += f" • {indicator_text}"
                        
                        st.write(header_text)
                        
                        # Message content
                        if message['type'] == 'emergency':
                            st.error(message['content'])
                        else:
                            st.write(message['content'])
                        
                        # Show sources if available
                        if 'metadata' in message and 'sources' in message['metadata']:
                            sources = message['metadata']['sources']
                            if isinstance(sources, list) and len(sources) > 1:
                                with st.expander("📚 Sources"):
                                    for source in sources:
                                        st.text(f"• {source}")

@log_function_call
def process_enhanced_audio_recording():
    """Enhanced audio recording with better UI and graceful fallback"""
    st.markdown("#### 🎙️ Voice Input")
    
    # Check if audio system is available
    if not hasattr(st.session_state, 'audio_processor') or not st.session_state.audio_processor:
        st.error("❌ Audio processor not initialized")
        return
    
    audio_info = st.session_state.audio_processor.get_audio_info()
    
    # Show audio system status
    if not audio_info['audio_system_working']:
        st.warning("""
        ⚠️ **Audio System Not Available**
        
        Microphone recording is not available in this environment. This commonly happens when:
        - Running in Docker containers without audio device access
        - No microphone hardware is available
        - Audio permissions are not granted
        
        **Alternative:** You can still type your questions in the text box below.
        """)
        
        # Show what features are available
        if audio_info['supported_features']:
            with st.expander("📋 Available Audio Features"):
                for feature in audio_info['supported_features']:
                    st.text(f"✅ {feature.replace('_', ' ').title()}")
        
        return
    
    # Audio system is working - show recording interface
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.is_recording:
            if st.button("🎙️ Start Recording", key="record_btn", type="primary", use_container_width=True):
                st.session_state.is_recording = True
                st.rerun()
        else:
            st.warning("🔴 **Recording in progress...** Speak clearly about your medical question")
            
            if st.button("⏹️ Stop & Transcribe", key="stop_btn", type="secondary", use_container_width=True):
                with st.spinner("🎙️ Processing your audio..."):
                    try:
                        audio_file = st.session_state.audio_processor.record_audio(duration=15)
                        
                        if audio_file:
                            transcription_result = st.session_state.audio_processor.transcribe_audio(audio_file)
                            
                            if transcription_result['success']:
                                st.session_state.transcribed_text = transcription_result['text']
                                st.success("✅ Audio transcribed successfully!")
                                st.info(f"**Transcribed:** \"{st.session_state.transcribed_text}\"")
                            else:
                                st.error(f"❌ Transcription failed: {transcription_result['error']}")
                        else:
                            st.error("❌ Recording failed!")
                    except Exception as e:
                        logger.error(f"Audio recording error: {str(e)}")
                        st.error(f"❌ Audio processing error: {str(e)}")
                
                st.session_state.is_recording = False
                st.rerun()

@log_function_call
def process_enhanced_user_query(query_text):
    """Enhanced query processing with Lambda integration"""
    if not query_text.strip():
        st.warning("Please provide a medical question.")
        return
    
    if not st.session_state.selected_therapeutic_area:
        st.warning("Please select a therapeutic area first.")
        return
    
    # Add user message to chat
    add_message_to_chat('user', query_text)
    
    # Process with enhanced AI including Lambda integration
    with st.spinner("🧠 Analyzing your question..."):
        try:
            response_data = st.session_state.rag_processor.process_medical_query(
                query=query_text,
                therapeutic_area=st.session_state.selected_therapeutic_area,
                use_lambda=st.session_state.use_lambda_integration
            )
            
            if response_data['success']:
                message_type = 'emergency' if response_data.get('emergency', False) else 'normal'
                add_message_to_chat(
                    'assistant', 
                    response_data['response'], 
                    message_type,
                    response_data.get('metadata', {})
                )
                
                # Show processing details
                with st.expander("📊 Processing Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Response Details:**")
                        st.write(f"- Confidence: {response_data.get('confidence', 0):.1%}")
                        st.write(f"- Processing Time: {response_data.get('processing_time', 0):.2f}s")
                        st.write(f"- Sources: {', '.join(response_data.get('sources', []))}")
                        
                        if response_data.get('metadata', {}).get('lambda_used'):
                            st.write("- 🔧 Lambda Enhanced: Yes")
                        else:
                            st.write("- 🔧 Lambda Enhanced: No")
                    
                    with col2:
                        if response_data.get('contexts'):
                            st.write(f"**Knowledge Sources ({len(response_data['contexts'])} found):**")
                            for i, context in enumerate(response_data['contexts'][:3], 1):
                                st.text(f"{i}. {context[:100]}...")
                        
                        # Show Lambda contribution if available
                        if response_data.get('lambda_contribution'):
                            st.write("**Lambda Contribution:**")
                            st.text(response_data['lambda_contribution'][:200] + "...")
                
                # Show confidence warning if needed
                confidence = response_data.get('confidence', 0.8)
                if confidence < 0.7:
                    st.warning(f"⚠️ Response confidence: {confidence:.1%} - Consider consulting additional resources.")
                
            else:
                error_response = f"I apologize, but I'm having technical difficulties. Error: {response_data.get('error', 'Unknown error')}"
                add_message_to_chat('assistant', error_response)
                
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            error_response = f"I'm currently experiencing technical issues. Please try again later."
            add_message_to_chat('assistant', error_response)

@log_function_call
def render_sidebar():
    """Render the enhanced sidebar with Lambda integration status"""
    with st.sidebar:
        st.markdown("### 🛠️ System Status")
        
        # System status checks
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        if aws_access_key:
            st.success("✅ AWS Credentials")
        else:
            st.error("❌ AWS Credentials")
        
        # AWS connection test
        try:
            if hasattr(st.session_state, 'rag_processor') and st.session_state.rag_processor:
                test_connection = st.session_state.rag_processor.test_aws_connection()
                if test_connection:
                    st.success("✅ AWS Connection")
                else:
                    st.error("❌ AWS Connection")
            else:
                st.error("❌ AWS Connection")
        except:
            st.error("❌ AWS Connection")
        
        # Lambda function status
        if st.session_state.lambda_status is not None:
            if st.session_state.lambda_status:
                st.success("✅ Lambda Function")
            else:
                st.warning("⚠️ Lambda Limited")
        else:
            st.info("ℹ️ Lambda Testing...")
        
        # Audio system status
        try:
            if hasattr(st.session_state, 'audio_processor') and st.session_state.audio_processor:
                audio_info = st.session_state.audio_processor.get_audio_info()
                
                if audio_info['audio_system_working']:
                    st.success("✅ Audio System")
                elif audio_info['audio_libraries_available']:
                    st.warning("⚠️ Audio (Limited)")
                else:
                    st.error("❌ Audio System")
            else:
                st.error("❌ Audio System")
        except Exception as e:
            st.error("❌ Audio System")
        
        st.markdown("---")
        
        # Lambda Integration Controls
        st.markdown("### 🔧 Lambda Integration")
        
        if st.session_state.lambda_status:
            st.success("wonderscribeconnectVDB: Active")
        else:
            st.warning("wonderscribeconnectVDB: Limited")
        
        st.session_state.use_lambda_integration = st.checkbox(
            "Enable Lambda Enhancement",
            value=st.session_state.use_lambda_integration,
            help="Use Lambda function for enhanced processing"
        )
        
        if st.button("🔄 Test Lambda", use_container_width=True):
            with st.spinner("Testing Lambda..."):
                lambda_working, lambda_message = test_lambda_connection()
                st.session_state.lambda_status = lambda_working
                if lambda_working:
                    st.success("Lambda test successful!")
                else:
                    st.error(f"Lambda test failed: {lambda_message}")
        
        st.markdown("---")
        
        # Quick learning modules
        st.markdown("### 📚 Quick Modules")
        if st.session_state.selected_therapeutic_area:
            area_info = THERAPEUTIC_AREAS[st.session_state.selected_therapeutic_area]
            for module in area_info.get('quick_modules', [])[:5]:  # Limit to 5
                if st.button(f"📖 {module}", key=f"module_{module}"):
                    st.session_state.transcribed_text = f"Tell me about {module} in {area_info['name']}"
                    st.rerun()
        
        st.markdown("---")
        
        # Session Statistics
        if st.session_state.chat_history:
            st.markdown("### 📈 Session Stats")
            if st.session_state.session_analytics:
                stats = st.session_state.session_analytics.get_session_stats(st.session_state.chat_history)
                
                st.metric("Questions", stats.get('total_questions', 0))
                st.metric("Avg Response Time", f"{stats.get('avg_response_time', 0):.1f}s")
                
                # Count Lambda-enhanced responses
                lambda_enhanced = sum(1 for msg in st.session_state.chat_history 
                                    if msg.get('lambda_enhanced', False))
                if lambda_enhanced > 0:
                    st.metric("Lambda Enhanced", lambda_enhanced)

@log_function_call
def main():
    """Main application function with Lambda integration"""
    logger.info("🚀 Starting Interactive Medical AI Coach with RAG Integration")
    
    # Initialize session state FIRST
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Interactive Medical AI Coach</h1>
        <p>Advanced Medical Education • Powered by AI & RAG Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Lambda integration status
    render_lambda_status()
    
    # Session dashboard
    render_session_dashboard()
    
    # Therapeutic area selection
    if not render_therapeutic_area_selector():
        st.info("👆 Please select a therapeutic area to begin your learning session.")
        return
    
    # Chat interface
    st.markdown("### 💬 Interactive Learning Session")
    display_enhanced_chat_history()
    
    # Input section
    st.markdown("### ✍️ Ask Your Question")
    
    # Text input with conditional voice option
    audio_available = (hasattr(st.session_state, 'audio_processor') and 
                      st.session_state.audio_processor and 
                      st.session_state.audio_processor.get_audio_info()['audio_system_working'])
    
    if audio_available:
        col1, col2 = st.columns([4, 1])
    else:
        col1, col2 = st.columns([1, 1])  # Full width for text input
    
    with col1:
        if st.session_state.transcribed_text:
            user_input = st.text_area(
                "Your medical question:", 
                value=st.session_state.transcribed_text,
                placeholder="Ask about diagnoses, treatments, mechanisms of action, clinical guidelines...",
                height=100,
                key="text_input"
            )
        else:
            user_input = st.text_area(
                "Your medical question:", 
                placeholder="Ask about diagnoses, treatments, mechanisms of action, clinical guidelines...",
                height=100,
                key="text_input"
            )
    
    # Only show voice input button if audio is available
    if audio_available:
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎙️ Voice Input", type="secondary"):
                st.session_state.show_voice_input = not st.session_state.get('show_voice_input', False)
                st.rerun()
    
    # Voice input section
    if st.session_state.get('show_voice_input', False):
        process_enhanced_audio_recording()
        
        if st.session_state.transcribed_text:
            if st.button("🗑️ Clear Transcription"):
                st.session_state.transcribed_text = ""
                st.rerun()
    
    # Send button with Lambda status indicator
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        lambda_indicator = "🔧" if st.session_state.use_lambda_integration and st.session_state.lambda_status else "📤"
        button_text = f"Send {lambda_indicator}"
        
        if st.button(button_text, type="primary", use_container_width=True):
            if user_input.strip():
                process_enhanced_user_query(user_input.strip())
                st.session_state.transcribed_text = ""
                st.rerun()
    
    # Session controls
    st.markdown("### 🔧 Session Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("📊 Export Session"):
            if st.session_state.session_analytics:
                session_data = st.session_state.session_analytics.export_session(
                    st.session_state.session_id,
                    st.session_state.chat_history
                )
                st.download_button(
                    "📄 Download Session Report",
                    session_data,
                    f"medical_ai_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
    
    with col3:
        if st.button("🔄 New Session"):
            st.session_state.chat_history = []
            st.session_state.session_start_time = datetime.now()
            st.session_state.session_id = f"session_{int(time.time())}"
            st.rerun()

if __name__ == "__main__":
    # Initialize session state first, then render sidebar, then run main
    if 'initialized' not in st.session_state:
        initialize_session_state()
        st.session_state.initialized = True
    
    # Render sidebar (now that session state is initialized)
    render_sidebar()
    
    # Run main application
    main()