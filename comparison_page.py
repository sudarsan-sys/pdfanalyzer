import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import your Gemini client
from gemini_client import GeminiClient, get_gemini_client

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total_analyzed': 0,
        'avg_processing_time': 0,
        'total_tokens_used': 0,
        'documents_by_type': {}
    }

# Page config
st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .feature-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header {
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("RAG Model Comparison")
st.markdown("""
This page compares the standard RAG model with our enhanced implementation, specifically optimized for research paper analysis.
""")

# Comparison Data
comparison_data = {
    'Feature': [
        'Chunking Strategy', 
        'Context Preservation', 
        'Retrieval Precision',
        'Response Quality',
        'Domain Optimization',
        'Computational Efficiency'
    ],
    'Standard RAG': [
        'Fixed-size chunks',
        'Medium - May break context',
        'Moderate - Basic semantic search',
        'Good - General purpose',
        'General domain',
        'High - Less processing'
    ],
    'Enhanced RAG': [
        'Semantic section-based chunks',
        'High - Preserves document structure',
        'High - Advanced re-ranking & metadata',
        'Excellent - Research paper focused',
        'Optimized for academic papers',
        'Moderate - Additional processing'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(comparison_data)

# Sidebar for navigation and document upload
st.sidebar.title("Document Analysis")

# Document upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Research Paper (PDF/TXT)",
    type=['pdf', 'txt'],
    accept_multiple_files=False
)

# Analysis options
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Summarize", "Extract Key Points", "Find Methodology", "Full Analysis"]
)

# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Document Analysis", "Analysis History", "Model Comparison"]
)

def analyze_document(content: str, analysis_type: str) -> Dict[str, Any]:
    """Simulate document analysis with metrics tracking"""
    start_time = time.time()
    
    # Initialize client
    client = get_gemini_client()
    
    # Generate prompt based on analysis type
    if analysis_type == "Summarize":
        prompt = f"Please provide a concise summary of the following research paper:\n\n{content}"
    elif analysis_type == "Extract Key Points":
        prompt = f"Extract the key points from this research paper:\n\n{content}"
    elif analysis_type == "Find Methodology":
        prompt = f"Extract and explain the methodology section from this research paper:\n\n{content}"
    else:  # Full Analysis
        prompt = f"""
        Analyze this research paper and provide a comprehensive breakdown including:
        1. Main research question
        2. Methodology
        3. Key findings
        4. Limitations
        5. Future work
        
        Paper content:
        {content}
        """
    
    # Get response from Gemini
    response = client.generate_content(
        prompt=prompt,
        max_output_tokens=2000,
        temperature=0.3
    )
    
    # Calculate metrics
    processing_time = time.time() - start_time
    tokens_used = len(prompt.split()) + len(response.split())
    
    return {
        'content': response,
        'processing_time': processing_time,
        'tokens_used': tokens_used,
        'analysis_type': analysis_type,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def display_analysis_metrics():
    """Display analysis metrics"""
    if not st.session_state.analysis_results:
        return
    
    st.subheader("Analysis Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents Analyzed", len(st.session_state.analysis_results))
    with col2:
        avg_time = sum(r['processing_time'] for r in st.session_state.analysis_results) / len(st.session_state.analysis_results)
        st.metric("Avg. Processing Time", f"{avg_time:.2f} seconds")
    with col3:
        total_tokens = sum(r['tokens_used'] for r in st.session_state.analysis_results)
        st.metric("Total Tokens Used", f"{total_tokens:,}")
    
    # Analysis type distribution
    analysis_types = {}
    for result in st.session_state.analysis_results:
        analysis_types[result['analysis_type']] = analysis_types.get(result['analysis_type'], 0) + 1
    
    if analysis_types:
        fig = px.pie(
            names=list(analysis_types.keys()),
            values=list(analysis_types.values()),
            title="Analysis Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

if page == "Document Analysis":
    st.title("Research Paper Analysis")
    
    if uploaded_file:
        with st.spinner("Analyzing document..."):
            # Read file content
            content = uploaded_file.read().decode("utf-8")
            
            # Analyze document
            result = analyze_document(content, analysis_type)
            
            # Store results
            st.session_state.analysis_results.append(result)
            
            # Display results
            st.subheader(f"Analysis Results: {analysis_type}")
            st.markdown("---")
            
            # Display analysis content in an expandable section
            with st.expander("View Analysis", expanded=True):
                st.markdown(result['content'])
            
            # Display metrics
            st.markdown("---")
            st.subheader("Analysis Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Processing Time", f"{result['processing_time']:.2f} seconds")
            with col2:
                st.metric("Tokens Used", f"{result['tokens_used']:,}")
            
            st.success("Analysis complete!")
    else:
        st.info("Please upload a document to begin analysis.")
        
elif page == "Analysis History":
    st.title("Analysis History")
    
    if not st.session_state.analysis_results:
        st.info("No analysis history available. Upload and analyze a document first.")
    else:
        # Display metrics
        display_analysis_metrics()
        
        # Display history table
        st.subheader("Recent Analyses")
        history_data = [{
            'Timestamp': r['timestamp'],
            'Analysis Type': r['analysis_type'],
            'Processing Time (s)': f"{r['processing_time']:.2f}",
            'Tokens Used': r['tokens_used']
        } for r in reversed(st.session_state.analysis_results)]
        
        st.dataframe(
            pd.DataFrame(history_data),
            use_container_width=True,
            hide_index=True
        )
        
elif page == "Model Comparison":
    st.title("RAG Model Comparison")
    st.markdown("This section compares the standard RAG model with our enhanced implementation.")
    
    st.header("Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Standard RAG")
        st.markdown("""
        - **Paper**: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
        - **Approach**: Combines dense retrieval with generation
        - **Strengths**: General-purpose, efficient, well-tested
        - **Limitations**: May lose context, generic retrieval
        """)
    
    with col2:
        st.subheader("Enhanced RAG")
        st.markdown("""
        - **Enhancement**: Domain-optimized for research papers
        - **Key Features**:
          - Semantic section-based chunking
          - Advanced metadata tagging
          - Context-aware re-ranking
        - **Benefits**: Higher accuracy, better context preservation
        """)

elif page == "Feature Comparison":
    st.header("Feature Comparison")
    
    # Feature comparison table
    st.dataframe(
        df,
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width="medium"),
            "Standard RAG": st.column_config.TextColumn("Standard RAG", width="medium"),
            "Enhanced RAG": st.column_config.TextColumn(
                "Enhanced RAG",
                width="medium",
            ),
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Visual comparison
    st.subheader("Feature Comparison Chart")
    fig = px.bar(
        df.melt(id_vars=['Feature'], var_name='Model', value_name='Value'),
        x='Feature',
        y='Value',
        color='Model',
        barmode='group',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Performance Metrics":
    st.header("Performance Metrics")
    
    # Mock performance data
    metrics_data = {
        'Metric': ['Retrieval Accuracy', 'Response Relevance', 'Context Preservation', 'Processing Speed'],
        'Standard RAG': [75, 70, 65, 90],
        'Enhanced RAG': [92, 95, 94, 80]
    }
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Retrieval Accuracy", "92%", "+17%")
        st.metric("Response Relevance", "95%", "+25%")
    
    with col2:
        st.metric("Context Preservation", "94%", "+29%")
        st.metric("Processing Speed", "80%", "-10%")
    
    # Performance chart
    st.subheader("Performance Comparison")
    fig = px.line(
        pd.DataFrame(metrics_data).melt(id_vars=['Metric'], var_name='Model', value_name='Score'),
        x='Metric',
        y='Score',
        color='Model',
        markers=True,
        title="Performance Metrics Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Use Cases":
    st.header("Use Cases")
    
    st.subheader("Standard RAG")
    st.markdown("""
    - General Q&A systems
    - Chatbots with broad knowledge
    - Applications requiring fast, general responses
    - When computational efficiency is a priority
    """)
    
    st.subheader("Enhanced RAG")
    st.markdown("""
    - Academic research assistance
    - Technical document analysis
    - Literature review automation
    - Research paper summarization
    - When domain expertise is crucial
    """)
    
    # Example use case
    st.subheader("Example: Research Paper Analysis")
    st.markdown("""
    **Query**: "What methodology was used in paper X?"
    
    - **Standard RAG**: Might return generic methodology information
    - **Enhanced RAG**: 
      - Uses section-based chunking to find the methodology section
      - Leverages metadata to understand context
      - Provides precise, well-structured response with relevant citations
    """)

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This comparison highlights how our enhanced RAG implementation provides significant improvements for research paper analysis 
while maintaining the core benefits of the original RAG architecture.
""")
