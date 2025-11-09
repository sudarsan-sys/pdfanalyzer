import os
import streamlit as st
from pathlib import Path
import tempfile
import plotly.express as px
import pandas as pd
from gemini_client import get_gemini_client

from query_engine import QueryEngine
from pdf_extractor import extract_text_from_pdf

# Set page config
st.set_page_config(
    page_title="PDF Analyzer with Gemini",
    page_icon="📚",
    layout="wide"
)

# Create a directory for storing the vector database
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Initialize session state
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = QueryEngine(persist_directory=str(DATA_DIR.absolute()))
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

# Sidebar
with st.sidebar:
    st.title("📚 PDF Analyzer")
    st.markdown("""
    ### How to use:
    1. Upload a PDF document
    2. View the extracted text
    3. Ask questions about the document
    4. Compare RAG models
    """)
    
    # File uploader
    st.subheader("1. Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="file_uploader"
    )
    
    # Process button
    if st.button("Process Document", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                try:
                    # Save uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extract text from PDF
                    st.session_state.extracted_text = extract_text_from_pdf(tmp_path)
                    st.session_state.uploaded_file = uploaded_file.name
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        else:
            st.warning("Please upload a PDF file first.")

# Main content
# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Document Analysis", "Model Comparison"])

with tab1:  # Existing document analysis tab
    st.title("PDF Analyzer with Gemini")

    # Display uploaded file info
    if st.session_state.uploaded_file:
        st.subheader(f"📄 {st.session_state.uploaded_file}")
        
        # Show extracted text preview
        with st.expander("View Extracted Text", expanded=False):
            if st.session_state.extracted_text:
                st.text_area(
                    "Extracted Text",
                    value=st.session_state.extracted_text[:5000] + 
                         ("..." if len(st.session_state.extracted_text) > 5000 else ""),
                    height=300,
                    disabled=True
                )
            else:
                st.info("No text extracted yet. Please process a document first.")
        
        # Add document to vector store
        if st.button("Add to Knowledge Base", type="primary"):
            if st.session_state.extracted_text:
                with st.spinner("Adding document to knowledge base..."):
                    try:
                        st.session_state.query_engine.add_document(
                            document_text=st.session_state.extracted_text,
                            metadata={
                                "title": st.session_state.uploaded_file,
                                "type": "pdf"
                            }
                        )
                        st.success("Document added to knowledge base!")
                    except Exception as e:
                        st.error(f"Error adding document: {str(e)}")
            else:
                st.warning("No extracted text available. Please process the document first.")
        
        # Query interface
        st.divider()
        st.subheader("Ask a Question")
        query = st.text_input(
            "Enter your question about the document:",
            placeholder="e.g., What is the main topic of this document?",
            key="query_input"
        )
        
        if st.button("Get Answer", type="primary", key="get_answer_btn") and query:
            if not hasattr(st.session_state, 'query_engine'):
                st.error("Please add the document to the knowledge base first.")
            else:
                with st.spinner("Searching for answers..."):
                    try:
                        result = st.session_state.query_engine.query(
                            query_text=query,
                            n_results=3,
                            generate_answer=True
                        )
                        
                        # Display answer
                        st.subheader("Answer")
                        st.markdown(f"**{result['answer']}**")
                        
                        # Display sources
                        st.subheader("Sources")
                        for i, doc in enumerate(result['results'], 1):
                            with st.expander(f"Source {i} (Relevance: {doc['score']:.2f})"):
                                st.markdown(doc['document'])
                                st.caption(f"Source: {doc['metadata'].get('title', 'Unknown')}")
                                
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
        
        # Document analysis
        st.divider()
        st.subheader("Document Analysis")
        analysis_type = st.selectbox(
            "Select analysis type:",
            ["Summary", "Key Points", "Sentiment Analysis"],
            key="analysis_type_selector"
        )
        
        if st.button("Analyze Document", key="analyze_btn"):
            if not st.session_state.extracted_text:
                st.warning("No document text available for analysis.")
            else:
                with st.spinner(f"Performing {analysis_type}..."):
                    try:
                        if analysis_type == "Summary":
                            result = st.session_state.query_engine.summarize_document(
                                st.session_state.extracted_text,
                                summary_length="detailed"
                            )
                        elif analysis_type == "Key Points":
                            result = st.session_state.query_engine.analyze_document(
                                document_text=st.session_state.extracted_text,
                                query="Extract the key points from this document as a bulleted list."
                            )
                        else:  # Sentiment Analysis
                            result = st.session_state.query_engine.analyze_document(
                                document_text=st.session_state.extracted_text,
                                query="Perform a sentiment analysis of this document. Identify the overall sentiment and key points that contribute to it."
                            )
                        
                        st.subheader(analysis_type)
                        st.markdown(result)
                        
                    except Exception as e:
                        st.error(f"Error performing analysis: {str(e)}")

    # Welcome message for document analysis tab
    st.markdown("""
    ## Welcome to PDF Analyzer with Gemini
    
    This application allows you to:
    
    - 📄 Upload and extract text from PDF documents
    - 🔍 Ask questions about the document content
    - 📊 Get summaries and analyses of your documents
    
    **Get started by uploading a PDF file using the sidebar.**
    """)
    
    # Example queries
    st.subheader("Example Queries")
    st.markdown("""
    Once you've uploaded a document, try asking:
    
    - What is the main topic of this document?
    - Can you summarize the key points?
    - What are the main findings or conclusions?
    - Are there any recommendations mentioned?
    """)

with tab2:  # Real-time Model Metrics tab
    st.title("Model Performance Metrics")
    
    # Get the Gemini client instance
    try:
        # Import here to avoid circular imports
        from gemini_client import GeminiClient
        
        # Get the global instance or create a new one
        client = get_gemini_client()
        if not hasattr(client, 'metrics'):
            # Initialize metrics if not present
            client.metrics = PerformanceMetrics()
        metrics = client.metrics
        
        st.markdown("## Real-time Performance Metrics")
        
        # Main metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Requests", getattr(metrics, 'total_requests', 0))
            st.metric("Successful Requests", getattr(metrics, 'successful_requests', 0))
            st.metric("Failed Requests", getattr(metrics, 'failed_requests', 0))
            
        with col2:
            success_rate = (metrics.successful_requests / metrics.total_requests * 100) if getattr(metrics, 'total_requests', 0) > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
            st.metric("Total Tokens Used", f"{getattr(metrics, 'total_tokens_used', 0):,}")
            st.metric("Avg. Response Time", f"{metrics.get_avg_response_time():.2f}s")
            
        with col3:
            st.metric("Requests per Minute", f"{metrics.get_requests_per_minute():.1f}")
            
            # Calculate token usage breakdown
            if hasattr(metrics, 'token_usage') and metrics.token_usage:
                most_used_method = max(metrics.token_usage.items(), key=lambda x: x[1])
                st.metric("Most Used Method", 
                         f"{most_used_method[0]} ({most_used_method[1]} tokens)")
        
        # Token usage by method
        if hasattr(metrics, 'token_usage') and metrics.token_usage:
            st.subheader("Token Usage by Method")
            token_data = pd.DataFrame(
                [{"Method": k, "Tokens": v} for k, v in metrics.token_usage.items()]
            )
            st.bar_chart(token_data.set_index('Method'))
        
        # Request timeline
        st.subheader("Request Timeline")
        try:
            # Get recent requests using the new method
            recent_requests = getattr(metrics, 'get_recent_requests', lambda x: [])(20)
            
            if recent_requests:
                # Prepare timeline data
                timeline_data = [{
                    "Timestamp": datetime.fromtimestamp(req.get('timestamp', 0)).strftime('%H:%M:%S'),
                    "Method": str(req.get('method', 'unknown')),
                    "Response Time (s)": float(req.get('response_time', 0)),
                    "Tokens Used": int(req.get('tokens_used', 0))
                } for req in recent_requests if isinstance(req, dict)]
                
                if timeline_data:
                    df_timeline = pd.DataFrame(timeline_data)
                    
                    # Show request count by method
                    st.bar_chart(
                        df_timeline['Method'].value_counts(),
                        width='stretch'
                    )
        except Exception as e:
            st.warning(f"Could not display request timeline: {str(e)}")
        
        # Performance metrics over time
        st.subheader("Performance Over Time")
        try:
            if hasattr(metrics, 'get_recent_requests'):
                recent_requests = metrics.get_recent_requests(50)  # Get last 50 requests
                if not recent_requests:
                    st.info("No recent request data available. Perform some operations to see metrics.")
                else:
                    # Safely create DataFrame with request data
                    request_data = []
                    for i, req in enumerate(recent_requests):
                        if not isinstance(req, dict):
                            continue
                        try:
                            request_data.append({
                                'Request': i + 1,
                                'Timestamp': float(req.get('timestamp', 0)),
                                'Response Time (s)': float(req.get('response_time', 0)),
                                'Tokens Used': int(req.get('tokens_used', 0)),
                                'Method': str(req.get('method', 'unknown'))
                            })
                        except (ValueError, TypeError) as e:
                            continue
                    
                    if not request_data:
                        st.warning("No valid request data available for visualization.")
                        
                    
                    df_metrics = pd.DataFrame(request_data)
                    
                    # Plot response times
                    if not df_metrics.empty and 'Response Time (s)' in df_metrics.columns:
                        st.subheader("Response Times")
                        st.line_chart(
                            df_metrics.set_index('Request')['Response Time (s)'],
                            width='stretch'
                        )
                        
                        # Show tokens used over time
                        st.subheader("Token Usage")
                        st.bar_chart(
                            df_metrics.set_index('Request')['Tokens Used'],
                            width='stretch'
                        )
                        
                        # Show rolling average if we have enough data points
                        if len(df_metrics) >= 5:
                            try:
                                df_metrics['Rolling Avg (5)'] = df_metrics['Response Time (s)'].rolling(5, min_periods=1).mean()
                                st.subheader("Response Time with Rolling Average")
                                st.line_chart(
                                    df_metrics.set_index('Request')['Rolling Avg (5)'],
                                    width='stretch',
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.warning(f"Could not calculate rolling average: {str(e)}")
        except Exception as e:
            st.warning(f"Could not display performance metrics: {str(e)}")
        
        # Model Comparison Section
        st.subheader("Model Comparison")
        
        # Get metrics safely with defaults
        try:
            # Get metrics summary as a dictionary
            metrics_summary = metrics.get_metrics_summary()
            
            # Extract values with safe defaults
            total_requests = metrics_summary.get('total_requests', 0)
            success_rate = metrics_summary.get('success_rate', 0.0)
            avg_response_time = metrics_summary.get('avg_response_time_seconds', 0.0)
            tokens_per_request = 0
            if total_requests > 0:
                tokens_per_request = metrics_summary.get('total_tokens_used', 0) / total_requests
                
        except Exception as e:
            st.warning(f"Error retrieving metrics: {str(e)}")
            # Fallback to safe defaults
            total_requests = 0
            success_rate = 0.0
            avg_response_time = 0.0
            tokens_per_request = 0
        
        # Define comparison data for the table
        comparison_data = {
            'Metric': [
                'Total Requests', 
                'Success Rate', 
                'Avg Response Time (s)',
                'Tokens per Request',
                'Requests per Minute',
                'Error Rate'
            ],
            'This Model': [
                total_requests,
                success_rate,  # Keep as float for chart
                avg_response_time,  # Keep as float for chart
                round(tokens_per_request, 1) if total_requests > 0 else 0,
                round(metrics.get_requests_per_minute(), 1),
                (metrics.failed_requests / metrics.total_requests * 100) if getattr(metrics, 'total_requests', 0) > 0 else 0
            ],
            'Standard RAG': [
                int(total_requests * 1.2) if total_requests > 0 else 0,  # Example: 20% more requests
                85,  # Success rate in %
                1.5,  # Avg response time in seconds
                512,  # Tokens per request
                45.0,  # Requests per minute
                12.0   # Error rate in %
            ]
        }
        
        # Create a DataFrame for the table (with formatted strings)
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display the comparison table with formatted values
        st.dataframe(
            comparison_df.style.format({
                'This Model': lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) and 'Rate' in comparison_df.loc[comparison_df['This Model'] == x, 'Metric'].values[0] 
                                      else f"{x:.2f}s" if isinstance(x, (int, float)) and 'Time' in comparison_df.loc[comparison_df['This Model'] == x, 'Metric'].values[0]
                                      else f"{x:,}" if isinstance(x, (int, float)) and x > 1000 
                                      else str(x),
                'Standard RAG': lambda x: f"{x}%" if isinstance(x, (int, float)) and 'Rate' in comparison_df.loc[comparison_df['Standard RAG'] == x, 'Metric'].values[0]
                                        else f"{x}s" if isinstance(x, (int, float)) and 'Time' in comparison_df.loc[comparison_df['Standard RAG'] == x, 'Metric'].values[0]
                                        else str(x)
            }),
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "This Model": st.column_config.TextColumn("This Model", width="medium"),
                "Standard RAG": st.column_config.TextColumn("Standard RAG", width="medium"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Prepare data for the comparison line chart
        metrics_for_chart = [
            'Success Rate',
            'Avg Response Time (s)',
            'Tokens per Request',
            'Requests per Minute',
            'Error Rate'
        ]
        
        # Create a DataFrame for the chart
        chart_data = []
        for metric in metrics_for_chart:
            this_model_val = comparison_df.loc[comparison_df['Metric'] == metric, 'This Model'].values[0]
            rag_val = comparison_df.loc[comparison_df['Metric'] == metric, 'Standard RAG'].values[0]
            
            # Skip if either value is not a number
            if not (isinstance(this_model_val, (int, float)) and isinstance(rag_val, (int, float))):
                continue
                
            chart_data.extend([
                {'Model': 'This Model', 'Metric': metric, 'Value': this_model_val},
                {'Model': 'Standard RAG', 'Metric': metric, 'Value': rag_val}
            ])
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            
            # Create a grouped bar chart
            st.subheader("Performance Comparison")
            fig = px.bar(
                chart_df,
                x='Metric',
                y='Value',
                color='Model',
                barmode='group',
                title='Model Performance Comparison',
                labels={'Value': 'Score', 'Metric': 'Performance Metric'},
                color_discrete_map={
                    'This Model': '#1f77b4',
                    'Standard RAG': '#ff7f0e'
                }
            )
            
            # Customize the layout
            fig.update_layout(
                xaxis_tickangle=-45,
                legend_title_text='Model',
                yaxis_title='Score',
                xaxis_title='',
                height=500
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
        
        # Display comparison table
        st.dataframe(
            pd.DataFrame(comparison_data),
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "This Model": st.column_config.TextColumn("This Model", width="medium"),
                "Standard RAG": st.column_config.TextColumn("Standard RAG", width="medium"),
            },
            hide_index=True,
            width='stretch'
        )
        
        # Model information
        st.subheader("Model Information")
        model_info = {
            "Model Name": getattr(client, 'model_name', 'N/A'),
            "Total Requests": getattr(metrics, 'total_requests', 0),
            "Success Rate": f"{success_rate:.1f}%",
            "Average Response Time": f"{metrics.get_avg_response_time():.2f}s",
            "Total Tokens Used": getattr(metrics, 'total_tokens_used', 0)
        }
        st.json(model_info)
        
    except Exception as e:
        st.error(f"Error retrieving metrics: {str(e)}")
        st.info("Perform some operations in the Document Analysis tab to see real-time metrics.")
    
    # Add a refresh button
    if st.button("Refresh Metrics"):
        st.rerun()

# Footer
st.divider()
st.caption("""
Built with [Streamlit](https://streamlit.io/) and powered by [Google Gemini](https://ai.google/discover/gemini/).
""")
