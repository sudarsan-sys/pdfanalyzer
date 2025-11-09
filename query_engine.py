import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json

from vector_store import ChromaDBStore
from embeddings import get_embedding_model
from gemini_client import get_gemini_client
from text_processor import TextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryEngine:
    """Orchestrates the query pipeline for document analysis."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        """Initialize the query engine.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to store the vector database
        """
        # Initialize the vector store
        self.vector_store = ChromaDBStore(
            collection_name=collection_name,
            persist_directory=persist_directory or "./chroma_db"
        )
        self.gemini = get_gemini_client()
        self.text_processor = TextProcessor()
    
    def add_document(
        self,
        document_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> None:
        """Add a document to the vector store.
        
        Args:
            document_text: Text content of the document
            metadata: Optional metadata for the document
            chunk: Whether to chunk the document before adding
        """
        if not document_text:
            return
            
        try:
            if chunk:
                # Process and chunk the document
                chunks = self.text_processor.chunk_text(
                    document_text,
                    clean=True,
                    metadata=metadata or {}
                )
                
                # Add each chunk to the vector store
                self.vector_store.add_documents(chunks)
            else:
                # Add as a single document
                doc = {'text': document_text}
                if metadata:
                    doc.update(metadata)
                self.vector_store.add_documents([doc])
                
            logger.info(f"Added document to vector store: {metadata.get('title', 'Untitled')}")
            
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """Query the knowledge base.
        
        Args:
            query_text: The query text
            n_results: Number of relevant chunks to retrieve
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_output_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Get relevant chunks from the vector store
            results = self.vector_store.search(query_text, n_results=n_results)
            
            if not results:
                return {
                    "response": "No relevant information found in the knowledge base.",
                    "sources": [],
                    "metadata": {}
                }
            
            # Format the context from search results
            context_parts = []
            for i, result in enumerate(results, 1):
                source = result.get('metadata', {}).get('source', f'Result {i}')
                context_parts.append(f"--- {source} ---\n{result.get('text', '')}")
            
            context = "\n\n".join(context_parts)
            
            # Generate a response using the Gemini model
            prompt = f"""You are a helpful AI assistant. Answer the question based on the following context. If you don't know the answer, say you don't know.
            
Context:
{context}

Question: {query_text}

Answer:"""
            
            # Remove generate_answer from kwargs as it's not a valid parameter for generate_content
            kwargs.pop('generate_answer', None)
            
            response = self.gemini.generate_content(
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                **kwargs
            )
            
            # Extract sources from results
            sources = list({result.get('metadata', {}).get('source', 'Unknown') 
                          for result in results if result.get('metadata', {}).get('source')})
            
            return {
                "answer": response,  # Changed from "response" to "answer" to match app.py expectations
                "results": [
                    {
                        "document": result.get('text', ''),
                        "metadata": result.get('metadata', {}),
                        "score": result.get('score', 0.0)
                    }
                    for result in results
                ],
                "sources": sources,
                "metadata": {
                    "n_results": len(results),
                    "model": self.gemini.model_name,
                    **{k: v for k, v in kwargs.items() if k != 'generate_answer'}
                }
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    def analyze_document(
        self,
        document_text: str,
        query: str,
        chunk_size: int = 4000,
        overlap: int = 200
    ) -> str:
        """Analyze a document by processing it with Gemini.
        
        Args:
            document_text: The document text to analyze
            query: The question or analysis request
            chunk_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            Analysis result as a string
        """
        return self.gemini.analyze_document(
            document_text=document_text,
            query=query,
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    def summarize_document(
        self,
        document_text: str,
        summary_length: str = "brief"  # 'brief', 'detailed', or 'comprehensive'
    ) -> str:
        """Generate a summary of a document.
        
        Args:
            document_text: The document text to summarize
            summary_length: Desired length of the summary
            
        Returns:
            Generated summary
        """
        length_prompt = {
            "brief": "a concise 2-3 sentence summary",
            "detailed": "a detailed paragraph",
            "comprehensive": "a comprehensive summary with key points"
        }.get(summary_length.lower(), "a summary")
        
        prompt = f"""Please provide {length_prompt} of the following document:

{document_text}

Summary:"""
        
        return self.gemini.generate_content(prompt)
    
    def clear_database(self) -> None:
        """Clear the vector database."""
        try:
            self.vector_store.delete_collection()
            logger.info("Vector database cleared")
        except Exception as e:
            logger.error(f"Failed to clear database: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    from pdf_extractor import extract_text_from_pdf
    
    # Initialize the query engine
    engine = QueryEngine(collection_name="test_collection")
    
    try:
        # Example: Add a document
        sample_text = """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
        the natural intelligence displayed by animals including humans. AI research has been defined as the 
        field of study of intelligent agents, which refers to any system that perceives its environment and 
        takes actions that maximize its chance of achieving its goals.
        
        Machine learning is a subfield of AI that focuses on building systems that learn from data. These systems 
        improve their performance on a specific task as they are exposed to more data over time."""
        
        print("Adding sample document...")
        engine.add_document(
            document_text=sample_text,
            metadata={"title": "AI and Machine Learning", "source": "sample"}
        )
        
        # Example: Query the document
        print("\nQuerying the document...")
        result = engine.query(
            "What is artificial intelligence?",
            generate_answer=True
        )
        
        print("\nGenerated Answer:")
        print(result['answer'])
        print("\nSource Documents:")
        for i, doc in enumerate(result['results'], 1):
            print(f"\nSource {i} (Score: {doc['score']:.4f}):")
            print(doc['document'][:200] + "...")
        
        # Example: Generate a summary
        print("\nGenerating a summary...")
        summary = engine.summarize_document(sample_text, "brief")
        print("Summary:", summary)
        
    finally:
        # Clean up
        engine.clear_database()
        print("\nTest collection deleted.")
