import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from utils.medical_rag import MedicalRAGProcessor
from utils.logging import setup_logging

# Try to import plotly with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available. Install plotly for advanced charts: `pip install plotly`")

# Setup logging
logger = setup_logging()

st.set_page_config(page_title="System Monitoring", page_icon="📊", layout="wide")

# Enhanced CSS for monitoring dashboard
st.markdown("""
<style>
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    
    .metric-card.healthy {
        border-left: 4px solid #22c55e;
    }
    
    .metric-card.warning {
        border-left: 4px solid #f59e0b;
    }
    
    .metric-card.critical {
        border-left: 4px solid #dc2626;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        text-align: center;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .status-healthy {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-degraded {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-critical {
        background: #fecaca;
        color: #991b1b;
    }
    
    .status-unknown {
        background: #f3f4f6;
        color: #374151;
    }
    
    .monitoring-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def get_status_badge(status: str) -> str:
    """Generate HTML status badge"""
    status_classes = {
        'healthy': 'status-healthy',
        'degraded': 'status-degraded', 
        'slow': 'status-degraded',
        'critical': 'status-critical',
        'unknown': 'status-unknown',
        'idle': 'status-unknown'
    }
    
    status_icons = {
        'healthy': '✅',
        'degraded': '⚠️',
        'slow': '🐌', 
        'critical': '🚨',
        'unknown': '❓',
        'idle': '💤'
    }
    
    css_class = status_classes.get(status, 'status-unknown')
    icon = status_icons.get(status, '❓')
    
    return f'<span class="status-badge {css_class}">{icon} {status.title()}</span>'

def render_system_overview():
    """Render system overview section"""
    st.markdown("""
    <div class="monitoring-header">
        <h1>📊 System Monitoring Dashboard</h1>
        <p>Lambda Performance • System Health • Real-time Metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status overview
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Initialize RAG processor if not already done
        if 'rag_processor' not in st.session_state:
            st.session_state.rag_processor = MedicalRAGProcessor()
        
        # Get system metrics
        lambda_metrics = st.session_state.rag_processor.get_lambda_metrics()
        aws_connection = st.session_state.rag_processor.test_aws_connection()
        
        with col1:
            aws_status = "✅ Connected" if aws_connection else "❌ Disconnected"
            st.metric("AWS Connection", aws_status)
        
        with col2:
            if lambda_metrics.get('monitoring_available'):
                lambda_health = lambda_metrics.get('health_status', 'unknown')
                lambda_status = f"{get_status_badge(lambda_health)}"
                st.markdown(f"**Lambda Health**<br>{lambda_status}", unsafe_allow_html=True)
            else:
                st.metric("Lambda Health", "⚠️ Limited")
        
        with col3:
            # Document processing status
            doc_chunks = len(st.session_state.rag_processor.document_chunks) if hasattr(st.session_state.rag_processor, 'document_chunks') else 0
            st.metric("Document Chunks", f"{doc_chunks:,}")
        
        with col4:
            # Session info
            if 'session_start_time' in st.session_state:
                uptime = datetime.now() - st.session_state.session_start_time
                uptime_str = f"{uptime.total_seconds() / 60:.0f}m"
            else:
                uptime_str = "N/A"
            st.metric("Session Uptime", uptime_str)
            
    except Exception as e:
        st.error(f"❌ Failed to load system overview: {str(e)}")

def render_lambda_metrics():
    """Render Lambda function metrics"""
    st.markdown("### 🔧 Lambda Function Metrics")
    
    try:
        if 'rag_processor' not in st.session_state:
            st.warning("⚠️ RAG processor not initialized")
            return
        
        lambda_metrics = st.session_state.rag_processor.get_lambda_metrics()
        
        if not lambda_metrics.get('monitoring_available'):
            st.info("ℹ️ Lambda monitoring not available. This is normal if the lambda_monitor module is not installed.")
            return
        
        if 'error' in lambda_metrics:
            st.error(f"❌ Error getting Lambda metrics: {lambda_metrics['error']}")
            return
        
        # Performance summary
        summary = lambda_metrics.get('summary', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_calls = summary.get('total_invocations', 0)
            st.metric("Total Invocations", f"{total_calls:,}")
        
        with col2:
            error_rate = summary.get('overall_error_rate', 0) * 100
            st.metric("Error Rate", f"{error_rate:.1f}%")
        
        with col3:
            avg_response = summary.get('avg_response_time', 0)
            st.metric("Avg Response Time", f"{avg_response:.2f}s")
        
        with col4:
            healthy_functions = summary.get('healthy_functions', 0)
            total_functions = summary.get('total_functions', 0)
            st.metric("Healthy Functions", f"{healthy_functions}/{total_functions}")
        
        # Function-specific metrics
        st.markdown("#### wonderscribeconnectVDB Function Details")
        
        wonderscribe_metrics = lambda_metrics.get('wonderscribe_function', {})
        
        if wonderscribe_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Performance Metrics**")
                
                metrics_data = [
                    ["Invocations", f"{wonderscribe_metrics.get('invocation_count', 0):,}"],
                    ["Success Rate", f"{wonderscribe_metrics.get('success_rate', 0):.1%}"],
                    ["Avg Duration", f"{wonderscribe_metrics.get('avg_duration', 0):.2f}s"],
                    ["Min Duration", f"{wonderscribe_metrics.get('min_duration', 0):.2f}s"],
                    ["Max Duration", f"{wonderscribe_metrics.get('max_duration', 0):.2f}s"],
                ]
                
                metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Status Information**")
                
                health_status = wonderscribe_metrics.get('health_status', 'unknown')
                st.markdown(f"**Health Status:** {get_status_badge(health_status)}", unsafe_allow_html=True)
                
                last_invocation = wonderscribe_metrics.get('last_invocation')
                if last_invocation:
                    last_time = datetime.fromisoformat(last_invocation.replace('Z', '+00:00'))
                    time_ago = datetime.now() - last_time.replace(tzinfo=None)
                    st.write(f"**Last Invocation:** {time_ago.total_seconds() / 60:.0f}m ago")
                
                last_error = wonderscribe_metrics.get('last_error')
                if last_error:
                    st.write(f"**Last Error:** {last_error[:100]}...")
        else:
            st.info("ℹ️ No metrics available for wonderscribeconnectVDB function yet. Make some queries to see metrics.")
        
        # Recent calls
        st.markdown("#### Recent Lambda Calls")
        
        recent_calls = lambda_metrics.get('recent_calls', [])
        
        if recent_calls:
            calls_df = pd.DataFrame(recent_calls)
            calls_df['timestamp'] = pd.to_datetime(calls_df['timestamp'])
            calls_df['duration'] = calls_df['duration'].round(2)
            calls_df['status'] = calls_df['success'].apply(lambda x: '✅ Success' if x else '❌ Failed')
            
            # Display table
            display_df = calls_df[['timestamp', 'function_name', 'duration', 'status', 'error']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            display_df.columns = ['Time', 'Function', 'Duration (s)', 'Status', 'Error']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Performance chart (only if plotly is available)
            if PLOTLY_AVAILABLE and len(calls_df) > 1:
                st.markdown("#### Performance Trend")
                
                fig = px.line(calls_df, x='timestamp', y='duration', 
                            title='Lambda Response Time Trend',
                            labels={'duration': 'Duration (seconds)', 'timestamp': 'Time'})
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            elif not PLOTLY_AVAILABLE and len(calls_df) > 1:
                st.markdown("#### Performance Trend")
                st.line_chart(calls_df.set_index('timestamp')['duration'])
        else:
            st.info("ℹ️ No recent Lambda calls to display.")
            
    except Exception as e:
        st.error(f"❌ Failed to load Lambda metrics: {str(e)}")
        logger.exception("Lambda metrics error:")

def render_document_processing_status():
    """Render document processing status"""
    st.markdown("### 📚 Document Processing Status")
    
    try:
        if 'rag_processor' not in st.session_state:
            st.warning("⚠️ RAG processor not initialized")
            return
        
        processor = st.session_state.rag_processor
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document Information**")
            
            # Document chunks
            if hasattr(processor, 'document_chunks') and processor.document_chunks:
                chunk_count = len(processor.document_chunks)
                avg_chunk_size = sum(chunk['char_count'] for chunk in processor.document_chunks) / chunk_count
                total_chars = sum(chunk['char_count'] for chunk in processor.document_chunks)
                
                st.metric("Total Chunks", f"{chunk_count:,}")
                st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} chars")
                st.metric("Total Characters", f"{total_chars:,}")
            else:
                st.warning("⚠️ No document chunks available")
        
        with col2:
            st.markdown("**Embedding Information**")
            
            # TF-IDF embeddings
            if hasattr(processor, 'tfidf_matrix') and processor.tfidf_matrix is not None:
                st.success(f"✅ TF-IDF: {processor.tfidf_matrix.shape}")
            else:
                st.warning("⚠️ TF-IDF embeddings not available")
            
            # Semantic embeddings
            if hasattr(processor, 'semantic_embeddings') and processor.semantic_embeddings is not None:
                st.success(f"✅ Semantic: {processor.semantic_embeddings.shape}")
            else:
                st.info("ℹ️ Semantic embeddings not available (optional)")
            
            # AWS S3 document info
            s3_bucket = processor.s3_bucket
            s3_key = processor.s3_document_key
            st.info(f"📁 Source: s3://{s3_bucket}/{s3_key}")
        
        # Document chunk distribution (simplified chart)
        if hasattr(processor, 'document_chunks') and processor.document_chunks:
            st.markdown("#### Document Chunk Distribution")
            
            chunk_sizes = [chunk['char_count'] for chunk in processor.document_chunks]
            
            if PLOTLY_AVAILABLE:
                fig = px.histogram(x=chunk_sizes, nbins=20, 
                                 title='Document Chunk Size Distribution',
                                 labels={'x': 'Chunk Size (characters)', 'y': 'Count'})
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to simple chart
                chunk_df = pd.DataFrame({'chunk_size': chunk_sizes})
                st.bar_chart(chunk_df['chunk_size'])
            
    except Exception as e:
        st.error(f"❌ Failed to load document processing status: {str(e)}")

def render_system_logs():
    """Render recent system logs"""
    st.markdown("### 📋 Recent System Logs")
    
    try:
        # Read recent log files
        log_files = [
            "logs/debug_" + datetime.now().strftime("%Y-%m-%d") + ".log",
            "logs/errors_" + datetime.now().strftime("%Y-%m-%d") + ".log",
            "logs/aws_operations_" + datetime.now().strftime("%Y-%m-%d") + ".log"
        ]
        
        logs_found = False
        for log_file in log_files:
            if os.path.exists(log_file):
                logs_found = True
                with st.expander(f"📄 {os.path.basename(log_file)}"):
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            # Show last 50 lines
                            recent_lines = lines[-50:] if len(lines) > 50 else lines
                            log_content = ''.join(recent_lines)
                            st.text(log_content)
                    except Exception as e:
                        st.error(f"Failed to read log file: {e}")
        
        if not logs_found:
            st.info("ℹ️ No log files found for today.")
            
    except Exception as e:
        st.error(f"❌ Failed to load system logs: {str(e)}")

def render_export_controls():
    """Render export and control options"""
    st.markdown("### 🔧 Controls & Exports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data Export**")
        
        if st.button("📊 Export Lambda Metrics", type="primary"):
            try:
                if 'rag_processor' in st.session_state:
                    filepath = st.session_state.rag_processor.export_lambda_metrics()
                    if filepath:
                        st.success(f"✅ Metrics exported to: {filepath}")
                        
                        # Offer download
                        if os.path.exists(filepath):
                            with open(filepath, 'r') as f:
                                st.download_button(
                                    "📥 Download Metrics",
                                    f.read(),
                                    file_name=os.path.basename(filepath),
                                    mime="application/json"
                                )
                    else:
                        st.error("❌ Failed to export metrics")
                else:
                    st.error("❌ RAG processor not available")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    with col2:
        st.markdown("**System Controls**")
        
        if st.button("🔄 Refresh Metrics"):
            st.rerun()
        
        if st.button("🧹 Clear Lambda Metrics"):
            try:
                if 'rag_processor' in st.session_state:
                    st.info("ℹ️ Lambda metrics cleared (restart session to fully reset)")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Clear failed: {str(e)}")
    
    with col3:
        st.markdown("**Health Checks**")
        
        if st.button("🔍 Test AWS Connection"):
            try:
                if 'rag_processor' in st.session_state:
                    connection_ok = st.session_state.rag_processor.test_aws_connection()
                    if connection_ok:
                        st.success("✅ AWS connection healthy")
                    else:
                        st.error("❌ AWS connection failed")
                else:
                    st.error("❌ RAG processor not available")
            except Exception as e:
                st.error(f"❌ Connection test failed: {str(e)}")
        
        if st.button("🧪 Test Lambda Function"):
            try:
                if 'rag_processor' in st.session_state:
                    test_payload = {
                        'api_Path': 'getStory',
                        'story_theme': 'test query',
                        'story_type': 'test',
                        'main_character': 'Test',
                        'story_lang': 'English',
                        'word_count': '50'
                    }
                    
                    result = st.session_state.rag_processor.call_lambda_function(test_payload)
                    
                    if result['success']:
                        st.success("✅ Lambda function responding")
                    else:
                        st.error(f"❌ Lambda test failed: {result.get('error', 'Unknown error')}")
                else:
                    st.error("❌ RAG processor not available")
            except Exception as e:
                st.error(f"❌ Lambda test failed: {str(e)}")

def main():
    """Main monitoring dashboard"""
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Show plotly status
    if not PLOTLY_AVAILABLE:
        st.sidebar.warning("⚠️ Plotly not installed - some charts unavailable")
    
    # Render dashboard sections
    render_system_overview()
    
    st.markdown("---")
    render_lambda_metrics()
    
    st.markdown("---")
    render_document_processing_status()
    
    st.markdown("---")
    render_system_logs()
    
    st.markdown("---")
    render_export_controls()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #64748b; margin-top: 2rem;">
        <p>📊 Medical AI Coach Monitoring Dashboard</p>
        <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Plotly Status: {"✅ Available" if PLOTLY_AVAILABLE else "❌ Not Available"}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()