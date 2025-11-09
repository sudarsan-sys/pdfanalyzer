import os
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class PerformanceMetrics:
    """Class to track performance metrics for API calls."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_response_time: float = 0.0
    _request_data: List[Dict[str, Any]] = field(default_factory=list)  # List of dicts with request data
    token_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))  # method_name -> tokens
    
    def add_request(self, method_name: str, tokens_used: int, response_time: float, success: bool = True):
        """Record a new API request.
        
        Args:
            method_name: Name of the method being called
            tokens_used: Number of tokens used in the request
            response_time: Time taken for the request in seconds
            success: Whether the request was successful
        """
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        self.total_tokens_used += tokens_used
        self.total_response_time += response_time
        
        # Store request data in a more structured way
        self._request_data.append({
            'timestamp': time.time(),
            'method': method_name,
            'tokens_used': tokens_used,
            'response_time': response_time,
            'success': success
        })
        
        # Update token usage by method
        self.token_usage[method_name] = self.token_usage.get(method_name, 0) + tokens_used
    
    @property
    def request_timestamps(self) -> List[Tuple[float, str]]:
        """Get request timestamps and methods in a safe format."""
        result = []
        for req in self._request_data:
            try:
                if isinstance(req, dict) and 'timestamp' in req and 'method' in req:
                    timestamp = float(req['timestamp'])
                    method = str(req['method'])
                    result.append((timestamp, method))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid request data: {req}, error: {e}")
                continue
        return result
        
    def get_recent_requests(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent requests."""
        return self._request_data[-limit:] if self._request_data else []
    
    def get_avg_response_time(self) -> float:
        """Get average response time in seconds."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    def get_requests_per_minute(self) -> float:
        """Calculate requests per minute based on the last hour."""
        if not self.request_timestamps:
            return 0.0
            
        now = time.time()
        last_hour = [ts for ts, _ in self.request_timestamps if now - ts <= 3600]
        if not last_hour:
            return 0.0
            
        time_span = (now - min(ts for ts, _ in last_hour)) / 60  # in minutes
        return len(last_hour) / max(1, time_span)  # Avoid division by zero
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        try:
            # Safely calculate metrics
            total_requests = int(getattr(self, 'total_requests', 0))
            successful_requests = int(getattr(self, 'successful_requests', 0))
            failed_requests = int(getattr(self, 'failed_requests', 0))
            total_tokens_used = int(getattr(self, 'total_tokens_used', 0))
            
            # Calculate success rate safely
            success_rate = 0.0
            if total_requests > 0:
                success_rate = (successful_requests / total_requests) * 100
            
            # Get other metrics with safe defaults
            avg_response_time = float(self.get_avg_response_time())
            requests_per_minute = float(self.get_requests_per_minute())
            
            # Safely convert token usage to dict
            token_usage = {}
            if hasattr(self, 'token_usage'):
                try:
                    token_usage = dict(self.token_usage)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error processing token usage: {e}")
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate,
                "total_tokens_used": total_tokens_used,
                "avg_response_time_seconds": avg_response_time,
                "requests_per_minute": requests_per_minute,
                "token_usage_by_method": token_usage
            }
            
        except Exception as e:
            logger.error(f"Error in get_metrics_summary: {str(e)}")
            # Return a safe default if anything goes wrong
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "total_tokens_used": 0,
                "avg_response_time_seconds": 0.0,
                "requests_per_minute": 0.0,
                "token_usage_by_method": {}
            }

class GeminiClient:
    """Client for interacting with the Gemini API with performance tracking."""
    
    DEFAULT_MODEL = "gemini-2.5-flash"  # Using a valid model from the available models list
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: str = DEFAULT_MODEL
    ):
        """Initialize the Gemini client with performance tracking.
        
        Args:
            api_key: Google AI API key. If not provided, will use GEMINI_API_KEY from .env
            model_name: Name of the Gemini model to use (default: gemini-2.5-flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY in .env or pass as argument."
            )
            
        self.model_name = model_name
        self.metrics = PerformanceMetrics()
        
        # Configure the API key
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"Initialized Gemini client with model: {model_name}")
    
    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = 2048,
        track_metrics: bool = True,
        **generation_config
    ) -> str:
        """Generate content using the Gemini model with performance tracking.
        
        Args:
            prompt: The prompt to generate content from
            temperature: Controls randomness (0.0 to 1.0)
            max_output_tokens: Maximum number of tokens to generate
            track_metrics: Whether to track performance metrics for this call
            **generation_config: Additional generation parameters
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                
        Returns:
            Generated text
        """
        start_time = time.time()
        response = None
        tokens_used = 0
        
        try:
            # Prepare generation config
            config = {
                'temperature': temperature,
                'max_output_tokens': max_output_tokens,
                **generation_config
            }
            
            # Generate content using the model's generate_content method
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(**config)
            )
            
            # Calculate tokens used (approximate)
            tokens_used = len(prompt.split()) + (max_output_tokens or 0)
            
            # Extract text from response
            result = None
            if hasattr(response, 'text'):
                result = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                result = response.candidates[0].content.parts[0].text
            elif hasattr(response, 'result') and hasattr(response.result, 'text'):
                result = response.result.text
            else:
                # Fallback to string representation
                result = str(response)
            
            # Update metrics if tracking is enabled
            if track_metrics:
                response_time = time.time() - start_time
                self.metrics.add_request(
                    method_name="generate_content",
                    tokens_used=tokens_used,
                    response_time=response_time,
                    success=True
                )
                
            return result
                
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            logger.error(f"Response object: {response}" if response else "No response object")
            
            # Update metrics for failed request
            if track_metrics:
                response_time = time.time() - start_time
                self.metrics.add_request(
                    method_name="generate_content",
                    tokens_used=tokens_used,
                    response_time=response_time,
                    success=False
                )
            raise
            
    # Alias for backward compatibility
    generate_text = generate_content
    
    def chat(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate a chat response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0.0 to 1.0)
            **kwargs: Additional model parameters
            
        Returns:
            Generated response
        """
        try:
            # Convert messages to the format expected by the API
            chat = self.model.start_chat(history=[])
            
            # Add all messages to the chat history
            for msg in messages[:-1]:
                if msg["role"].lower() == "user":
                    chat.send_message(msg["content"])
                else:
                    # For assistant messages, we need to add them as a response
                    chat.history.append({
                        "role": "model",
                        "parts": [{"text": msg["content"]}]
                    })
            
            # Generate response for the last user message
            response = chat.send_message(
                messages[-1]["content"],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    **kwargs
                )
            )
            
            if not response.text:
                raise ValueError("No response was generated by the model")
                
            return response.text
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise

    def analyze_document(
        self,
        document_text: str,
        query: str,
        chunk_size: int = 4000,
        overlap: int = 200
    ) -> str:
        """Analyze a document by processing it in chunks.
        
        Args:
            document_text: The full text of the document.
            query: The question or analysis request.
            chunk_size: Maximum tokens per chunk.
            overlap: Number of tokens to overlap between chunks.
            
        Returns:
            Analysis result as a string.
        """
        # Simple chunking - in a real app, use the TextProcessor
        chunks = self._chunk_text(document_text, chunk_size, overlap)
        
        # Process each chunk and collect responses
        responses = []
        for i, chunk in enumerate(chunks, 1):
            chunk_prompt = f"""Document chunk {i}/{len(chunks)}:
{chunk}

Question: {query}"""
            
            response = self.generate_text(
                prompt=chunk_prompt,
                temperature=0.7,
                max_output_tokens=2048
            )
            responses.append(response)
        
        # Combine and summarize responses if needed
        if len(responses) > 1:
            combined = "\n\n".join(
                f"Chunk {i+1} analysis:\n{resp}" 
                for i, resp in enumerate(responses)
            )
            return self.generate_text(
                prompt=f"""Combine these analyses into a single coherent response:
                
{combined}

Question: {query}

Provide a comprehensive answer:""",
                temperature=0.7,
                max_output_tokens=2048,
                max_tokens=2048
            )
        return responses[0] if responses else "No analysis could be performed."
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple text chunking. In a real app, use the TextProcessor class."""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end == len(words):
                break
                
            # Move back by overlap to create overlap between chunks
            start = end - overlap
        
        return chunks

# Singleton instance
_gemini_client = None

def get_gemini_client(
    api_key: Optional[str] = None,
    model_name: str = GeminiClient.DEFAULT_MODEL
) -> GeminiClient:
    """Get or create a singleton instance of the Gemini client.
    
    Args:
        api_key: Optional API key. If None, will use environment variable.
        model_name: Name of the Gemini model to use.
        
    Returns:
        GeminiClient instance.
    """
    global _gemini_client
    
    if _gemini_client is None or _gemini_client.model_name != model_name:
        _gemini_client = GeminiClient(api_key=api_key, model_name=model_name)
    
    return _gemini_client

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.metrics.get_metrics_summary()

# Example usage
if __name__ == "__main__":
    try:
        # Initialize client
        client = GeminiClient()
        
        # Test performance metrics
        print("Testing performance metrics...")
        
        # Make some API calls
        for i in range(3):
            response = client.generate_content(
                f"Tell me a short fact about number {i+1}",
                max_output_tokens=100
            )
            print(f"Response {i+1}: {response[:100]}...")
        
        # Get and display metrics
        metrics = client.get_performance_metrics()
        print("\n=== Performance Metrics ===")
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Successful Requests: {metrics['successful_requests']}")
        print(f"Failed Requests: {metrics['failed_requests']}")
        print(f"Success Rate: {metrics['success_rate']:.2f}%")
        print(f"Total Tokens Used: {metrics['total_tokens_used']}")
        print(f"Average Response Time: {metrics['avg_response_time_seconds']:.2f}s")
        print(f"Requests per Minute: {metrics['requests_per_minute']:.2f}")
        print("\nToken Usage by Method:")
        for method, tokens in metrics['token_usage_by_method'].items():
            print(f"  - {method}: {tokens} tokens")
            
        # List available models
        print("Available models:")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f"- {m.name}")
        
        # Initialize client
        print("\nInitializing Gemini client...")
        client = get_gemini_client()
        
        # Test text generation
        print("\nTesting text generation...")
        response = client.generate_text("Tell me a short joke about AI")
        print(f"Response: {response}")
        
        # Example 2: Document analysis
        print("Testing document analysis...")
        sample_doc = """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
        the natural intelligence displayed by animals including humans. AI research has been defined as the 
        field of study of intelligent agents, which refers to any system that perceives its environment and 
        takes actions that maximize its chance of achieving its goals."""
        
        analysis = client.analyze_document(
            document_text=sample_doc,
            query="What is artificial intelligence according to this document?"
        )
        print("Analysis:", analysis)
        
    except Exception as e:
        print(f"Error: {str(e)}")
