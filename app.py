# Add these imports at the top of app.py
from semantic_chunker import SemanticChunker
import json
from typing import List, Dict, Any
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

def display_semantic_chunks(chunks: List[Dict[str, Any]]):
    """Display semantic chunks in an expandable format."""
    st.subheader("📑 Document Structure")
    
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"📌 {chunk.get('title', f'Section {i}')}"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("**Section Type**")
                st.info(chunk.get('section', 'Content'))
                
                if chunk.get('keywords'):
                    st.markdown("**Keywords**")
                    st.write(", ".join(chunk['keywords']))
                
                st.markdown("**Summary**")
                st.info(chunk.get('summary', 'No summary available'))
            
            with col2:
                st.markdown("**Content**")
                st.write(chunk.get('text', 'No content available'))
                
                if chunk.get('metadata', {}).get('context'):
                    with st.expander("View Context Analysis"):
                        context = chunk['metadata']['context']
                        st.json(context, expanded=False)

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
                    # Clear previous semantic chunks
                    if 'semantic_chunks' in st.session_state:
                        del st.session_state.semantic_chunks
                    
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

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Document Analysis", "Model Comparison", "Semantic Analysis"])

with tab1:  # Document Analysis tab
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
                            st.session_state.extracted_text,
                            metadata={"source": st.session_state.uploaded_file}
                        )
                        st.success("Document added to knowledge base!")
                    except Exception as e:
                        st.error(f"Error adding to knowledge base: {str(e)}")
            else:
                st.warning("No document text available to add to knowledge base.")

    # Welcome message
    else:
        st.markdown("""
        ## Welcome to PDF Analyzer with Gemini
        
        This application allows you to:
        - 📄 Upload and extract text from PDF documents
        - 🔍 Ask questions about the document content
        - 📊 Get summaries and analyses of your documents
        
        **Get started by uploading a PDF file using the sidebar.**
        """)

with tab2:  # Model Comparison tab
    st.title("Model Comparison")
    st.info("This tab will show comparison metrics between different models.")
    # Add your model comparison content here

with tab3:  # Semantic Analysis tab
    st.title("Semantic Document Analysis")
    
    if not st.session_state.get('extracted_text'):
        st.info("Please upload and process a document in the 'Document Analysis' tab first.")
    else:
        # Initialize semantic chunker if not already in session state
        if 'semantic_chunker' not in st.session_state:
            st.session_state.semantic_chunker = SemanticChunker()
        
        # Process document with semantic chunking
        if 'semantic_chunks' not in st.session_state:
            with st.spinner("Performing semantic analysis..."):
                try:
                    st.session_state.semantic_chunks = st.session_state.semantic_chunker.process_document(
                        st.session_state.extracted_text,
                        metadata={
                            'source': st.session_state.get('uploaded_file', 'unknown'),
                            'processed_at': str(pd.Timestamp.now())
                        }
                    )
                    st.success("Semantic analysis completed!")
                except Exception as e:
                    st.error(f"Error during semantic analysis: {str(e)}")
                    st.session_state.semantic_chunks = []
        
        # Display chunks if available
        if st.session_state.get('semantic_chunks'):
            display_semantic_chunks(st.session_state.semantic_chunks)
            
            # Q&A Section
            st.divider()
            st.subheader("🤖 Ask a Question")
            
            question = st.text_input(
                "Ask a question about the document:",
                placeholder="Type your question here...",
                key="question_input"
            )
            
            if question:
                with st.spinner("Analyzing document for answer..."):
                    # Simple implementation - in production, you'd want to use RAG
                    context = "\n\n".join(
                        f"## {chunk.get('title', 'Section')}\n{chunk.get('text', '')}"
                        for chunk in st.session_state.semantic_chunks
                    )
                    
                    prompt = f"""Answer the following question based on the document content.
                    If the answer cannot be found in the document, say so.
                    
                    QUESTION: {question}
                    
                    DOCUMENT CONTENT:
                    {context}
                    
                    Please provide a clear and concise answer.
                    """
                    
                    try:
                        response = get_gemini_client().generate_content(prompt)
                        st.markdown("### Answer")
                        st.write(response)
                        
                        # Show relevant context
                        with st.expander("View relevant context", expanded=False):
                            # Find most relevant chunks
                            relevant_chunks = sorted(
                                st.session_state.semantic_chunks,
                                key=lambda x: sum(
                                    1 for kw in x.get('keywords', [])
                                    if str(kw).lower() in question.lower()
                                ),
                                reverse=True
                            )[:2]  # Show top 2 most relevant chunks
                            
                            for chunk in relevant_chunks:
                                st.markdown(f"#### {chunk.get('title', 'Section')}")
                                st.caption(f"Keywords: {', '.join(map(str, chunk.get('keywords', [])))}")
                                st.write(chunk.get('summary', ''))
                                st.write("---")
                                
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

# Footer
st.divider()
st.caption("""
Built with [Streamlit](https://streamlit.io/) and powered by [Google Gemini](https://ai.google/discover/gemini/).
""")