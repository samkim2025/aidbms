import os
import time
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import Groq

class LLMHandler:
    def __init__(self):
        # Initialize API key
        api_key = st.secrets["GROQ_API_KEY"]
        if not api_key:
            raise ValueError("No GROQ_API_KEY found in secrets")
            
        # Configure Groq
        self.client = Groq(api_key=api_key)
        self.model = "mixtral-8x7b-32768"  # Groq's recommended model
        
        # Configuration
        self.chunk_size = 1000
        self.max_retries = 3

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=6)
    )
    def get_response(self, prompt: str) -> str:
        """Get response from Groq API"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant focused on document classification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100,
                top_p=0.8,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            st.write(f"API call failed: {str(e)}")
            raise

    def categorize_content(self, content: str) -> dict:
        try:
            # If content is too long, process in chunks
            if len(content) > self.chunk_size:
                chunks = [content[i:i+self.chunk_size] 
                         for i in range(0, len(content), self.chunk_size)]
                
                progress_text = "Analyzing document..."
                my_bar = st.progress(0)
                
                categories = []
                for idx, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks
                    my_bar.progress((idx + 1) / min(len(chunks), 5))
                    
                    prompt = f"""Analyze this text and respond with ONLY ONE category name. 
                    Text: {chunk}
                    Category:"""
                    
                    try:
                        result = self.get_response(prompt)
                        if result:
                            categories.append(result.strip())
                        time.sleep(1)  # Rate limiting
                    except Exception as e:
                        st.write(f"Chunk processing failed: {str(e)}")
                
                my_bar.empty()
                
                if categories:
                    # Get final category
                    final_prompt = f"""Given these categories: {', '.join(categories)}
                    Respond with the single most appropriate category name."""
                    
                    try:
                        final_category = self.get_response(final_prompt)
                        return {"category": final_category.strip()}
                    except Exception as e:
                        # Fallback to most common category
                        return {"category": max(set(categories), key=categories.count)}
                
                return {"category": "Uncategorized"}
                
            else:
                # For short content, process directly
                prompt = f"""Analyze this text and respond with ONLY ONE category name.
                Text: {content}
                Category:"""
                
                try:
                    result = self.get_response(prompt)
                    return {"category": result.strip() if result else "Uncategorized"}
                except Exception as e:
                    st.write(f"Categorization failed: {str(e)}")
                    return {"category": "Uncategorized"}
                    
        except Exception as e:
            st.write(f"Categorization error: {str(e)}")
            return {"category": "Uncategorized"}
