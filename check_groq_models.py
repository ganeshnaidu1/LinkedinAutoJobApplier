import os
import requests
from dotenv import load_dotenv

def list_available_models():
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env file")
        return
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            print("Available models:")
            for model in response.json()["data"]:
                print(f"- {model['id']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error making request: {str(e)}")

if __name__ == "__main__":
    list_available_models()
