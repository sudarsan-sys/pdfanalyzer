import os
import logging
from typing import List, Dict, Any, Optional, Union, Sequence
import numpy as np
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBStore:
    """A wrapper around ChromaDB for vector storage and retrieval."""
    
    def __init__(
        self, 
        collection_name: str = "documents", 
        persist_directory: str = "./chroma_db"
    ):
        """Initialize the ChromaDB store.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = os.path.abspath(persist_directory)
        
        # Create the client and collection
        self.client = self._get_client()
        self.collection = self._get_or_create_collection()
    
    def _get_client(self):
        """Create and return a ChromaDB client."""
        try:
            # Create the persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            return chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
    
    def _get_or_create_collection(self):
        """Get an existing collection or create a new one."""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as e:
            logger.error(f"Failed to get or create collection: {str(e)}")
            raise
    
    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the collection.
        
        Args:
            documents: List of document dictionaries with 'text' and other fields
            ids: Optional list of document IDs
            metadatas: Optional list of metadata dictionaries
        """
        if not documents:
            return
            
        try:
            # Extract texts
            texts = [doc.get('text', '') for doc in documents]
            
            # Generate or use provided IDs
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Prepare metadata if not provided
            if metadatas is None:
                metadatas = []
                for doc in documents:
                    metadata = doc.copy()
                    metadata.pop('text', None)
                    metadatas.append(metadata)
            
            # Upsert to collection (add or update)
            self.collection.upsert(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Upserted {len(documents)} documents to collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: The query string
            n_results: Number of results to return (default: 5)
            filter_conditions: Optional filter conditions
            
        Returns:
            List of result dictionaries with 'document', 'metadata', and 'score'
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_conditions
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1.0 - results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            return {
                'name': self.collection_name,
                'document_count': self.collection.count(),
                'embedding_dimension': 'unknown'  # Not directly available in the latest API
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            raise
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            raise
    
    def reset(self) -> None:
        """Reset the entire database."""
        try:
            self.client.reset()
            logger.info("Database reset complete")
        except Exception as e:
            logger.error(f"Failed to reset database: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create a test store
    store = ChromaDBStore(
        collection_name="test_collection"
    )
    
    # Add some test documents
    test_docs = [
        {"text": "This is a test document about artificial intelligence.", "source": "test"},
        {"text": "Machine learning is a subset of AI.", "source": "test"},
        {"text": "Neural networks are used in deep learning.", "source": "test"}
    ]
    
    store.add_documents(test_docs)
    
    # Search for similar documents
    results = store.search("What is AI?")
    
    print("Search results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['score']:.4f}):")
        print(f"Text: {result['document']}")
        print(f"Metadata: {result['metadata']}")
    
    # Clean up
    store.delete_collection()
