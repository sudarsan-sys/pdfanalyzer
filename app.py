import os
import streamlit as st
from pathlib import Path
import tempfile

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
        placeholder="e.g., What is the main topic of this document?"
    )
    
    if st.button("Get Answer", type="primary") and query:
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
        ["Summary", "Key Points", "Sentiment Analysis"]
    )
    
    if st.button("Analyze Document"):
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

else:
    # Welcome message
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

# Footer
st.divider()
st.caption("""
Built with [Streamlit](https://streamlit.io/) and powered by [Google Gemini](https://ai.google/discover/gemini/).
""")
