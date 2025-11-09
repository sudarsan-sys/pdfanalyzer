import os
import re
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

# Updated import for langchain text splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TextSplitter
)
from typing_extensions import Literal, TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingConfig(TypedDict, total=False):
    """Configuration for text chunking."""
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    keep_separator: bool
    is_separator_regex: bool

class TextProcessor:
    """
    A class for processing and chunking text documents.
    """
    
    DEFAULT_CONFIG: ChunkingConfig = {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'separators': ["\n\n", "\n", ". ", " ", ""],
        'keep_separator': True,
        'is_separator_regex': False
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TextProcessor with optional configuration.
        
        Args:
            config: Optional configuration dictionary. If not provided,
                   DEFAULT_CONFIG will be used.
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._splitter = self._create_splitter()
    
    def _create_splitter(self) -> TextSplitter:
        """Create a text splitter based on the current configuration."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            separators=self.config['separators'],
            keep_separator=self.config['keep_separator'],
            is_separator_regex=self.config['is_separator_regex']
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove non-printable characters except newlines and tabs
        text = re.sub(r'[^\x20-\x7E\t\n\r]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove excessive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def chunk_text(
        self,
        text: str,
        clean: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with optional cleaning.
        
        Args:
            text: Input text to chunk
            clean: Whether to clean the text before chunking
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text:
            return []
        
        # Clean the text if requested
        if clean:
            text = self.clean_text(text)
        
        # Split into chunks
        chunks = self._splitter.split_text(text)
        
        # Prepare result with metadata
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk,
                'chunk_num': i + 1,
                'total_chunks': len(chunks),
                **(metadata or {})
            }
            result.append(chunk_data)
        
        return result
    
    def chunk_document(
        self,
        file_path: str,
        encoding: str = 'utf-8',
        clean: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Read a text file and split it into chunks.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding (default: utf-8)
            clean: Whether to clean the text before chunking
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            
            metadata = {
                'source': str(Path(file_path).resolve()),
                'file_name': Path(file_path).name
            }
            
            return self.chunk_text(text, clean=clean, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

def create_default_processor() -> TextProcessor:
    """Create a TextProcessor with default settings."""
    return TextProcessor()

# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    processor = TextProcessor()
    
    sample_text = """
    This is a sample text document. It contains multiple paragraphs.
    
    Each paragraph is separated by blank lines. We'll split this into chunks.
    """
    
    chunks = processor.chunk_text(sample_text)
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print("-" * 40)
        print(chunk['text'])
    
    # Example 2: With custom configuration
    custom_config = {
        'chunk_size': 500,
        'chunk_overlap': 100
    }
    
    custom_processor = TextProcessor(custom_config)
    # ... use custom_processor as needed
