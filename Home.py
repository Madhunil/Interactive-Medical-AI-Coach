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

# Enhanced CSS for modern medical interface
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Lambda Integration Status */
    .lambda-status {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .lambda-status.enabled {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-color: #22c55e;
    }
    
    .lambda-status.disabled {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-color: #f59e0b;
    }
    
    /* Therapeutic Area Selector */
    .therapeutic-area-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .therapeutic-area-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
        text-align: center;
    }
    
    .therapeutic-area-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    
    .therapeutic-area-card.selected {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-color: #3b82f6;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
    }
    
    .area-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .area-name {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1e40af;
    }
    
    .area-description {
        font-size: 0.9rem;
        color: #64748b;
        line-height: 1.4;
    }
    
    /* Chat Container */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        background: linear-gradient(to bottom, #fafafa 0%, #f5f5f5 100%);
        margin-bottom: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
    }
    
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #a1a1a1;
    }
    
    /* Message Styles */
    .user-message {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 4px solid #22c55e;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.1);
        position: relative;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
        position: relative;
    }
    
    .lambda-enhanced-message {
        background: linear-gradient(135deg, #fdf4ff 0%, #f3e8ff 100%);
        border-left: 4px solid #8b5cf6;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.1);
        position: relative;
    }
    
    .emergency-message {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0 12px 12px 0;
        color: #dc2626;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.1);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 2px 8px rgba(220, 38, 38, 0.1); }
        50% { box-shadow: 0 4px 16px rgba(220, 38, 38, 0.2); }
    }
    
    /* Input Container */
    .input-container {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .input-container:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
    }
    
    /* Recording Indicator */
    .recording-indicator {
        background: linear-gradient(135deg, #fef3c7 0%, #fcd34d 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        animation: recording-pulse 1.5s infinite;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.2);
    }
    
    @keyframes recording-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Session Info Card */
    .session-info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Quick Actions */
    .quick-actions {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .quick-action-btn {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .quick-action-btn:hover {
        background: #e2e8f0;
        border-color: #94a3b8;
    }
    
    /* Sidebar Enhancements */
    .sidebar-section {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
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
    
    status_class = "enabled" if lambda_working else "disabled"
    status_icon = "✅" if lambda_working else "⚠️"
    status_text = "Lambda Integration Active" if lambda_working else "Lambda Integration Limited"
    
    st.markdown(f"""
    <div class="lambda-status {status_class}">
        <h4>{status_icon} {status_text}</h4>
        <p><strong>Function:</strong> wonderscribeconnectVDB</p>
        <p><strong>Status:</strong> {lambda_message}</p>
        <p><strong>API Gateway:</strong> {'Connected' if lambda_working else 'Limited'}</p>
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
            st.markdown(f"""
            <div class="session-info-card">
                <h4>{area_info['icon']} Current Area: {area_info['name']}</h4>
                <p>{area_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
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
        lambda_color = "normal" if st.session_state.lambda_status else "inverse"
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
    st.session_state.session_analytics.log_interaction(
        st.session_state.session_id,
        role,
        content,
        st.session_state.selected_therapeutic_area
    )

@log_function_call
def display_enhanced_chat_history():
    """Display enhanced chat history with Lambda integration indicators"""
    if not st.session_state.chat_history:
        welcome_area = st.session_state.selected_therapeutic_area
        area_info = THERAPEUTIC_AREAS.get(welcome_area, {'name': 'General Medicine', 'icon': '🩺'})
        
        lambda_info = ""
        if st.session_state.use_lambda_integration and st.session_state.lambda_status:
            lambda_info = "<br><br><strong>🔧 Lambda Integration:</strong> Enhanced processing with wonderscribeconnectVDB function enabled"
        
        st.markdown(f"""
        <div class="chat-container">
            <div class="ai-message">
                <strong>{area_info['icon']} Medical AI Coach - {area_info['name']}</strong><br>
                Welcome to the Interactive Medical AI Coach! I'm here to help you learn about {area_info['name'].lower()}. 
                You can ask questions using text or voice input.
                <br><br>
                <em>💡 Tip: Try asking about specific conditions, treatments, drug mechanisms, or clinical guidelines.</em>
                {lambda_info}
                <br><br>
                <strong>⚠️ Remember:</strong> This is for educational purposes only. Always follow clinical guidelines and consult healthcare professionals.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
        chat_html = '<div class="chat-container">'
        
        for message in st.session_state.chat_history:
            timestamp = datetime.fromtimestamp(message['timestamp']).strftime("%H:%M")
            
            if message['role'] == 'user':
                chat_html += f'''
                <div class="user-message">
                    <strong>👤 You • {timestamp}</strong><br>
                    {message['content']}
                </div>
                '''
            elif message['role'] == 'assistant':
                # Determine message style based on type and Lambda usage
                if message['type'] == 'emergency':
                    css_class = "emergency-message"
                    icon = "🚨"
                elif message.get('lambda_enhanced', False):
                    css_class = "lambda-enhanced-message"
                    icon = "🔧"
                else:
                    css_class = "ai-message"
                    icon = "🧠"
                
                # Add confidence and Lambda indicators
                indicators = []
                if 'metadata' in message and 'confidence' in message['metadata']:
                    confidence = message['metadata']['confidence']
                    if confidence < 0.7:
                        indicators.append(f"<span style='color: #f59e0b;'>⚠️ {confidence:.0%} confidence</span>")
                    else:
                        indicators.append(f"<span style='color: #22c55e;'>✅ {confidence:.0%} confidence</span>")
                
                if message.get('lambda_enhanced', False):
                    indicators.append("<span style='color: #8b5cf6;'>🔧 Lambda Enhanced</span>")
                
                if 'metadata' in message and 'sources' in message['metadata']:
                    sources = message['metadata']['sources']
                    if isinstance(sources, list) and len(sources) > 1:
                        indicators.append(f"<span style='color: #06b6d4;'>📚 {len(sources)} sources</span>")
                
                indicator_text = " • ".join(indicators)
                if indicator_text:
                    indicator_text = " • " + indicator_text
                
                chat_html += f'''
                <div class="{css_class}">
                    <strong>{icon} AI Coach • {timestamp}{indicator_text}</strong><br>
                    {message['content'].replace(chr(10), '<br>')}
                </div>
                '''
        
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

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
            st.markdown("""
            <div class="recording-indicator">
                🔴 <strong>Recording in progress...</strong><br>
                <em>Speak clearly about your medical question</em>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("⏹️ Stop & Transcribe", key="stop_btn", type="secondary", use_container_width=True):
                with st.spinner("🎙️ Processing your audio..."):
                    try:
                        audio_file = st.session_state.audio_processor.record_audio(duration=15)
                        
                        if audio_file:
                            transcription_result = st.session_state.audio_processor.transcribe_audio(audio_file)
                            
                            if transcription_result['success']:
                                st.session_state.transcribed_text = transcription_result['text']
                                st.success("✅ Audio transcribed successfully!")
                                
                                st.markdown(f"""
                                <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                                    <strong>Transcribed:</strong> "{st.session_state.transcribed_text}"
                                </div>
                                """, unsafe_allow_html=True)
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
                    if st.button("ℹ️ Audio Info", key="audio_info_btn"):
                        st.info("""
                        **Audio Status:** Limited functionality
                        - Audio libraries: Available
                        - Microphone: Not available
                        - Text-to-speech: """ + ("Available" if audio_info.get('aws_polly_available') else "Not available") + """
                        
                        This is normal in containerized environments.
                        """)
                else:
                    st.error("❌ Audio System")
            else:
                st.error("❌ Audio System")
        except Exception as e:
            st.error("❌ Audio System")
            logger.debug(f"Audio status check failed: {str(e)}")
        
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
            for module in area_info.get('quick_modules', []):
                if st.button(f"📖 {module}", key=f"module_{module}"):
                    st.session_state.transcribed_text = f"Tell me about {module} in {area_info['name']}"
                    st.rerun()
        
        st.markdown("---")
        
        # Session Statistics
        if st.session_state.chat_history:
            st.markdown("### 📈 Session Stats")
            stats = st.session_state.session_analytics.get_session_stats(st.session_state.chat_history)
            
            st.metric("Questions", stats.get('total_questions', 0))
            st.metric("Avg Response Time", f"{stats.get('avg_response_time', 0):.1f}s")
            
            # Count Lambda-enhanced responses
            lambda_enhanced = sum(1 for msg in st.session_state.chat_history 
                                if msg.get('lambda_enhanced', False))
            if lambda_enhanced > 0:
                st.metric("Lambda Enhanced", lambda_enhanced)
            
            if stats.get('topics_covered'):
                st.markdown("**Topics Covered:**")
                for topic in stats['topics_covered'][:5]:  # Show top 5
                    st.text(f"• {topic}")
        
        st.markdown("---")
        
        # Help and tips
        st.markdown("### 💡 Tips")
        st.markdown("""
        **🎯 Effective Questions:**
        - "Explain the mechanism of action of..."
        - "What are the side effects of..."
        - "Compare treatment options for..."
        - "What are the latest guidelines for..."
        
        **🔧 Lambda Integration:**
        - Enable for enhanced processing
        - Combines multiple AI models
        - Provides richer responses
        
        **🎙️ Voice Tips:**
        - Speak clearly and slowly
        - Mention specific drug names clearly
        - Ask one question at a time
        """)

@log_function_call
def main():
    """Main application function with Lambda integration"""
    logger.info("🚀 Starting Interactive Medical AI Coach with Lambda Integration")
    
    # Initialize session state FIRST
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Interactive Medical AI Coach</h1>
        <p>Advanced Medical Education • Powered by AI & Lambda Integration</p>
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
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
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
            if st.button("🎙️", help="Voice Input", type="secondary"):
                st.session_state.show_voice_input = not st.session_state.get('show_voice_input', False)
                st.rerun()
    else:
        # Show info about audio unavailability
        if col2:
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.button("🎙️", help="Voice input not available in this environment", disabled=True)
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Session controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("📊 Export Session"):
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