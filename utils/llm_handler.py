import os
import google.generativeai as genai
import streamlit as st

class LLMHandler:
    def __init__(self):
        # Debug logging
        st.write("Initializing LLM Handler...")
        
        # Try to get API key from secrets first
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.write("✅ Successfully retrieved API key from Streamlit secrets")
            # Verify the key isn't empty or malformed
            if not api_key or len(api_key) < 10:  # Basic validation
                st.write("⚠️ API key from secrets appears to be invalid")
                api_key = None
        except Exception as e:
            st.write(f"❌ Error accessing Streamlit secrets: {str(e)}")
            api_key = None

        # Fallback to environment variable if needed
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                st.write("✅ Using API key from environment variables")
            else:
                st.write("❌ No valid API key found in environment variables")

        # Final validation
        if not api_key:
            raise ValueError("No valid GOOGLE_API_KEY found in any location")
            
        # Configure Gemini
        try:
            # Print first and last 4 chars of key for debugging (safely)
            key_preview = f"{api_key[:4]}...{api_key[-4:]}"
            st.write(f"Attempting to configure Gemini with key: {key_preview}")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            
            # Test the configuration
            test_response = self.model.generate_content("Test")
            st.write("✅ Successfully tested Gemini API connection")
            
        except Exception as e:
            st.write(f"❌ Error configuring/testing Gemini: {str(e)}")
            raise

    def categorize_content(self, content: str) -> dict:
        try:
            response = self.model.generate_content(content)
            return {"category": response.text}
        except Exception as e:
            st.write(f"❌ Error in categorization: {str(e)}")
            return {"category": "Error", "error": str(e)}
