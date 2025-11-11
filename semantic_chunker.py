import json
from typing import List, Dict, Any
from gemini_client import get_gemini_client

class SemanticChunker:
    """
    A class for splitting documents into semantically meaningful chunks and extracting context.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the SemanticChunker.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        self.client = get_gemini_client(model_name=model_name)
    
    def _split_document_into_sections(self, document: str) -> List[Dict[str, Any]]:
        """
        Split a document into logical sections using LLM.
        
        Args:
            document: The document text to split
            
        Returns:
            List of dictionaries containing section text and metadata
        """
        prompt = f"""Split this document into logical sections with labels and summaries.
        For each section, identify:
        1. A short title (3-5 words)
        2. The section type (e.g., Introduction, Methodology, Results, etc.)
        3. A 1-2 sentence summary
        4. 3-5 keywords
        5. The section content

        Return the response as a valid JSON array of objects with these fields:
        - title: Section title
        - section: Section type/heading
        - summary: Brief summary
        - keywords: List of keywords
        - text: The actual section text

        Document:
        {document[:10000]}  # Limit to first 10k chars to avoid token limits
        """
        
        try:
            response = self.client.generate_content(prompt)
            
            # Handle different response formats
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'result') and hasattr(response.result, 'text'):
                content = response.result.text
            else:
                raise ValueError("Unexpected response format from Gemini API")
            
            # Clean up the response to ensure it's valid JSON
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
                
            sections = json.loads(content)
            if not isinstance(sections, list):
                sections = [sections]
            return sections
            
        except Exception as e:
            print(f"Error in _split_document_into_sections: {str(e)}")
            # Fallback to simple paragraph-based splitting if LLM fails
            return self._fallback_chunking(document)
    
    def _fallback_chunking(self, document: str) -> List[Dict[str, str]]:
        """
        Fallback chunking method when LLM fails.
        
        Splits document into paragraphs and creates simple chunks.
        """
        paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
        return [{
            'title': f"Section {i+1}",
            'section': 'Content',
            'summary': p[:200] + '...' if len(p) > 200 else p,
            'keywords': [],
            'text': p
        } for i, p in enumerate(paragraphs) if p.strip()]

    def extract_context(self, text: str) -> Dict[str, Any]:
        """
        Extract context from a chunk of text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing extracted context
        """
        try:
            prompt = f"""Analyze this text and extract the following:
            1. Main claims or key points (as a list)
            2. Key concepts (as a list)
            3. Any mentioned datasets or sources (as a list)
            
            Return the response as a JSON object with these fields:
            - claims: List of main claims
            - concepts: List of key concepts
            - datasets: List of mentioned datasets/sources
            
            Text to analyze:
            {text[:5000]}  # Limit to first 5k chars
            """
            
            response = self.client.generate_content(prompt)
            
            # Handle different response formats
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'result') and hasattr(response.result, 'text'):
                content = response.result.text
            else:
                raise ValueError("Unexpected response format from Gemini API")
                
            # Clean up the response
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
                
            return json.loads(content)
            
        except Exception as e:
            print(f"Error extracting context: {str(e)}")
            # Return empty context if extraction fails
            return {
                'claims': [],
                'concepts': [],
                'datasets': []
            }

    def process_document(self, document: str, metadata: dict = None) -> List[Dict[str, Any]]:
        """
        Process a document into semantic chunks with context.
        
        Args:
            document: The document text to process
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of chunks with semantic information and context
        """
        if not document or not document.strip():
            return []
            
        # Split document into sections
        sections = self._split_document_into_sections(document)
        
        # Add context to each section
        result = []
        for section in sections:
            if not isinstance(section, dict):
                continue
                
            # Ensure required fields exist
            section.setdefault('title', 'Untitled Section')
            section.setdefault('section', 'Content')
            section.setdefault('summary', '')
            section.setdefault('keywords', [])
            section.setdefault('text', '')
            
            # Extract context if we have text
            context = {}
            if section['text']:
                context = self.extract_context(section['text'])
            
            # Create the chunk with metadata and context
            chunk = {
                'title': section['title'],
                'section': section['section'],
                'summary': section['summary'],
                'keywords': section['keywords'],
                'text': section['text'],
                'metadata': {
                    **({'context': context} if context else {}),
                    **(metadata or {})
                }
            }
            result.append(chunk)
            
        return result