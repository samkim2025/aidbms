import json
from dotenv import load_dotenv
import os
import google.generativeai as genai
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Optional
import streamlit as st

class LLMHandler:
    def __init__(self):
        # Try to get API key from different sources
        api_key = (
            st.secrets.get("GOOGLE_API_KEY") or  # Streamlit secrets
            os.getenv("GOOGLE_API_KEY") or       # Environment variable
            None
        )
        
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in Streamlit secrets "
                "or environment variables."
            )
            
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.timeout = 30  # seconds
        self.max_retries = 3
        self.chunk_size = 2000  # Smaller chunks
        self.delay_between_calls = 2  # seconds
    
    def get_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            return "Uncategorized"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, Exception))
    )
    def _safe_generate(self, content: str) -> Optional[dict]:
        try:
            response = self.model.generate_content(
                content,
                generation_config={"temperature": 0.3, "max_output_tokens": 150}
            )
            time.sleep(self.delay_between_calls)  # Add delay between API calls
            return response
        except Exception as e:
            print(f"API call failed: {str(e)}")
            raise

    def categorize_content(self, content: str, progress_bar=None) -> dict:
        try:
            # If content is too long, process in chunks
            if len(content) > self.chunk_size:
                chunks = [content[i:i+self.chunk_size] 
                         for i in range(0, len(content), self.chunk_size)]
                results = []
                
                for i, chunk in enumerate(chunks):
                    if progress_bar:
                        progress_bar.progress((i + 1) / len(chunks))
                    
                    prompt = f"""Analyze this document chunk and provide key topics:
                    {chunk}
                    
                    Provide only the most relevant category."""
                    
                    result = self._safe_generate(prompt)
                    if result:
                        results.append(result)
                
                # Combine results
                final_prompt = f"""Based on these topics from document chunks:
                {[r.text for r in results if r]}
                
                What is the single most appropriate category?
                Respond with only the category name."""
                
                final_result = self._safe_generate(final_prompt)
                return {"category": final_result.text if final_result else "Uncategorized"}
            
            else:
                # Process short content directly
                prompt = f"""Analyze this document and provide the most appropriate category:
                {content}
                
                Respond with only the category name."""
                
                result = self._safe_generate(prompt)
                return {"category": result.text if result else "Uncategorized"}
                
        except Exception as e:
            print(f"Categorization failed: {str(e)}")
            return {"category": "Uncategorized", "error": str(e)}
