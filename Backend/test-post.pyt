import requests
import json

def send_message(message, qtt_type="basic", llm_model="openai"):
    """
    Send a message to the API endpoint
    
    Args:
        message (str): The query message
        qtt_type (str): Query type - "basic", "fusion", or "decomposition"
        llm_model (str): LLM model to use - "openai" or "gemini"
    """
    # API endpoint
    url = "http://localhost:3001/api/send-message"  # Adjust the base URL as needed
    
    # Request payload
    payload = {
        "message": message,
        "qtt": qtt_type,
        "llm_model": llm_model
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send POST request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return response data
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example 1: Basic query with OpenAI
    result1 = send_message(
        message="How many hours of fire resistance are required for stairs and elevators in non-combustible construction above 30 meters?",
        qtt_type="fusion",
        llm_model="gemini"
    )
    print("Gemini fusion Response:", result1)
    
