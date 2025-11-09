import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file")
    exit(1)

# Configure the API key
try:
    genai.configure(api_key=api_key)
    
    # List available models
    print("Available models:")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"- {m.name}")
    
    # Test text generation
    print("\nTesting text generation...")
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Tell me a short joke about AI")
    
    # Print the response
    if hasattr(response, 'text'):
        print(f"\nResponse: {response.text}")
    elif hasattr(response, 'candidates') and response.candidates:
        print(f"\nResponse: {response.candidates[0].content.parts[0].text}")
    else:
        print("\nError: Could not get response text")
        print(f"Response object: {response}")
    
except Exception as e:
    print(f"\nError: {str(e)}")
    print("\nMake sure you have the latest version of google-generativeai installed:")
    print("pip install --upgrade google-generativeai")
