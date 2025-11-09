import io
import logging
from pathlib import Path
from typing import Union, Optional, BinaryIO

import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract
from typing_extensions import Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors"""
    pass

def extract_text_from_pdf(
    source: Union[str, Path, BinaryIO],
    method: Literal["pdfplumber", "pdfminer"] = "pdfplumber",
    password: Optional[str] = None,
) -> str:
    """
    Extract text from a PDF file using the specified method.
    
    Args:
        source: Path to PDF file or file-like object
        method: Extraction method ('pdfplumber' or 'pdfminer')
        password: Password for encrypted PDFs (if any)
    
    Returns:
        Extracted text as a single string
    
    Raises:
        PDFExtractionError: If extraction fails
        FileNotFoundError: If PDF file doesn't exist
        ValueError: For invalid input types
    """
    try:
        if method == "pdfplumber":
            return _extract_with_pdfplumber(source, password)
        elif method == "pdfminer":
            return _extract_with_pdfminer(source, password)
        else:
            raise ValueError(f"Unsupported extraction method: {method}")
    
    except Exception as e:
        error_msg = f"Failed to extract text from PDF: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise PDFExtractionError(error_msg) from e

def _extract_with_pdfplumber(source: Union[str, Path, BinaryIO], password: Optional[str]) -> str:
    """Extract text using pdfplumber."""
    text_parts = []
    
    if isinstance(source, (str, Path)):
        with pdfplumber.open(source, password=password) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
    else:
        # Handle file-like objects
        source.seek(0)  # Ensure we're at the start of the file
        with pdfplumber.open(io.BytesIO(source.read()), password=password) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        source.seek(0)  # Reset file pointer
    
    if not text_parts:
        raise PDFExtractionError("No text could be extracted from the PDF")
    
    return "\n\n".join(text_parts)

def _extract_with_pdfminer(source: Union[str, Path, BinaryIO], password: Optional[str]) -> str:
    """Extract text using pdfminer.six (fallback method)."""
    try:
        if isinstance(source, (str, Path)):
            return pdfminer_extract(str(source), password=password or "")
        else:
            # Handle file-like objects
            source.seek(0)
            return pdfminer_extract(io.BytesIO(source.read()), password=password or "")
    finally:
        if hasattr(source, 'seek'):
            source.seek(0)

def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize the extracted text.
    
    Args:
        text: Raw extracted text from PDF
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize newlines
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    # Normalize unicode and other common issues
    text = text.replace('\x00', '')  # Remove null bytes
    text = ' '.join(text.split())  # Normalize all whitespace
    
    return text

# Example usage
if __name__ == "__main__":
    # Example 1: Extract from file path
    try:
        text = extract_text_from_pdf("example.pdf")
        print(f"Extracted {len(text)} characters")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Extract from file-like object
    with open("example.pdf", "rb") as f:
        try:
            text = extract_text_from_pdf(f)
            print(f"Extracted {len(text)} characters")
        except Exception as e:
            print(f"Error: {e}")
