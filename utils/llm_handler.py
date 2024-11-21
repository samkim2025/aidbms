import json
from dotenv import load_dotenv
import os
import google.generativeai as genai
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMHandler:
    def __init__(self):
        load_dotenv()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
        self.timeout = 30  # seconds
    
    def get_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            return "Uncategorized"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def categorize_content(self, content: str) -> dict:
        try:
            # Split content if it's too long
            max_length = 4000  # Adjust based on your needs
            if len(content) > max_length:
                content = content[:max_length] + "..."

            # Add timeout to the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}
                ],
                timeout=self.timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in categorization: {str(e)}")
            return {"category": "Uncategorized", "reason": "Processing error"}
