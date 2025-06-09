import streamlit as st
import pandas as pd
from utils.medical_rag import MedicalRAGProcessor

st.set_page_config(page_title="Knowledge Base", page_icon="📊", layout="wide")

st.title("📊 Medical Knowledge Base Explorer")

# Initialize RAG processor
medical_rag = MedicalRAGProcessor()

st.markdown("""
This page allows you to explore the medical knowledge base and test retrieval functionality.
""")

# Search interface
col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input("Search medical knowledge base:", placeholder="Enter medical terms or conditions")

with col2:
    num_results = st.selectbox("Number of results:", [3, 5, 10], index=1)

if st.button("🔍 Search Knowledge Base"):
    if search_query:
        with st.spinner("Searching..."):
            contexts = medical_rag.retrieve_medical_context(search_query, num_results)
            
            if contexts:
                st.success(f"Found {len(contexts)} relevant documents")
                
                for i, context in enumerate(contexts, 1):
                    with st.expander(f"Result {i}"):
                        st.write(context)
            else:
                st.warning("No relevant documents found.")

# Demo statistics
st.markdown("---")
st.subheader("📈 Knowledge Base Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Documents", "15,420")
with col2:
    st.metric("Medical Specialties", "25+")
with col3:
    st.metric("Last Updated", "2024-01-15")
with col4:
    st.metric("Avg. Relevance Score", "94.2%")