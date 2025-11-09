import os
import logging
import numpy as np
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EmbeddingModel:
    """A wrapper around Hugging Face's SentenceTransformer for generating embeddings."""
    
    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CACHE_DIR = os.path.join(str(Path.home()), ".cache", "huggingface", "transformers")
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        use_auth_token: bool = True
    ):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on (e.g., 'cuda' or 'cpu')
            use_auth_token: Whether to use Hugging Face auth token
        """
        self.model_name = model_name
        self.device = device or self._get_available_device()
        self.use_auth_token = use_auth_token
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                token=os.getenv("HF_TOKEN") if self.use_auth_token else None,
                cache_folder=self.CACHE_DIR
            )
            logger.info(f"Model loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def _get_available_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        if not text or not text.strip():
            return [0.0] * 384  # Return zero vector for empty text
            
        try:
            return self.model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 384  # Return zero vector on error
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process in a batch
            
        Returns:
            2D numpy array where each row is an embedding
        """
        if not texts:
            return np.array([])
            
        try:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True
            )
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return np.zeros((len(texts), 384))  # Return zero vectors on error
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if not hasattr(self, '_embedding_dim'):
            # Get dimension by encoding a dummy text
            dummy_embedding = self.get_embedding("test")
            self._embedding_dim = len(dummy_embedding)
        return self._embedding_dim

# Singleton instance
_embedding_model = None

def get_embedding_model(
    model_name: str = EmbeddingModel.DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    use_auth_token: bool = True
) -> 'EmbeddingModel':
    """Get or create a singleton instance of the embedding model.
    
    Args:
        model_name: Name of the Hugging Face model to use
        device: Device to run the model on (e.g., 'cuda' or 'cpu')
        use_auth_token: Whether to use Hugging Face auth token
        
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model
    
    if _embedding_model is None or _embedding_model.model_name != model_name:
        _embedding_model = EmbeddingModel(
            model_name=model_name,
            device=device,
            use_auth_token=use_auth_token
        )
    
    return _embedding_model

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the embedding model
        model = get_embedding_model()
        
        # Example 1: Get embedding for a single text
        text = "This is a sample sentence."
        embedding = model.get_embedding(text)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"Sample embedding (first 5 dims): {embedding[:5]}")
        
        # Example 2: Get embeddings for multiple texts
        texts = ["First text", "Second text", "Third text"]
        embeddings = model.get_embeddings(texts)
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"First embedding (first 5 dims): {embeddings[0][:5]}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
