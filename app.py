import streamlit as st
from utils.parser import DocumentParser
from utils.database import DatabaseHandler
from utils.llm_handler import LLMHandler
from utils.categorizer import AICategorizer
from utils.pdf_parser import PdfParser
from utils.text_parser import TextParser
from utils.docx_parser import DocxParser
import os
import threading
from functools import wraps
import time

st.set_page_config(page_title="Document Navigator", layout="wide")

# Initialize handlers
@st.cache_resource
def init_handlers():
    parser = DocumentParser()
    db_handler = DatabaseHandler()
    llm_handler = LLMHandler()
    categorizer = AICategorizer(llm_handler)
    return parser, db_handler, llm_handler, categorizer

parser, db_handler, llm_handler, categorizer = init_handlers()

def timeout_handler(timeout_duration=5):
    """Decorator to handle timeouts for functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_duration)
            
            if thread.is_alive():
                return None, "Operation timed out"
            if error[0] is not None:
                return None, str(error[0])
            return result[0], None
            
        return wrapper
    return decorator

@timeout_handler(5)  # 5 second timeout
def read_file_content(file):
    """Read content based on file type with timeout"""
    file_type = file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            content = PdfParser.parse(file)
            if not content:
                st.warning(f"Could not extract text from {file.name}. This might be a scanned document or protected PDF.")
            return content
        elif file_type == 'txt':
            return TextParser.parse(file)
        elif file_type == 'docx':
            return DocxParser.parse(file)
        else:
            return f"Unsupported file type: {file_type}"
    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
        return ""

def process_file(file, parser, db_handler, llm_handler, categorizer):
    try:
        with st.spinner(f'Processing {file.name}...'):
            # Parse content
            content = parser.parse(file)
            
            # Process in chunks if content is too long
            max_chunk = 4000
            if len(content) > max_chunk:
                chunks = [content[i:i+max_chunk] for i in range(0, len(content), max_chunk)]
                results = []
                for chunk in chunks:
                    result = llm_handler.categorize_content(chunk)
                    results.append(result)
                # Combine results (implement logic based on your needs)
                final_result = combine_results(results)
            else:
                final_result = llm_handler.categorize_content(content)
            
            # Store in database
            db_handler.store_document(file.name, content, final_result)
            
            return final_result
            
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def combine_results(results):
    # Implement logic to combine multiple categorization results
    # This is a simple example - adjust based on your needs
    categories = [r.get('category', 'Uncategorized') for r in results if r]
    # Return most common category or implement more sophisticated logic
    return {'category': max(set(categories), key=categories.count)}

@timeout_handler(5)
def categorize_file(content, categories):
    """Categorize file content with timeout"""
    if not categories or not content.strip():
        return "Uncategorized"
    
    prompt = f"""
    You are an expert document classifier with broad knowledge across multiple disciplines. 
    Categorize the following document into exactly ONE of these categories: {', '.join(categories)}

    Classification Guidelines:
    1. Content Analysis:
       - Consider the main subject matter and themes
       - Look for specialized terminology
       - Consider academic and technical context
       - Evaluate the overall focus, not just specific terms
    
    2. Decision Making:
       - Choose the category that best matches the primary topic
       - Consider both explicit and implicit subject matter
       - If content spans multiple categories, select the most dominant theme
       - Use "Uncategorized" only if no category clearly fits
    
    Document excerpt:
    ---
    {content[:3000]}
    ---

    Analysis steps:
    1. Identify the main topic and themes
    2. Consider the document's context and purpose
    3. Match with the most appropriate category

    Available categories: {', '.join(categories)}
    Return ONLY the category name:"""
    
    try:
        response = llm_handler.get_response(prompt)
        if isinstance(response, tuple):
            return "Uncategorized"
        
        response = response.strip()
        # Case-insensitive category matching
        for category in categories:
            if response.lower() == category.lower():
                return category
        return "Uncategorized"
    except Exception as e:
        print(f"Categorization error: {str(e)}")
        return "Uncategorized"

def main():
    st.title("AI Document Management System")
    
    # Add instructions
    st.markdown("""
    ### Instructions:
    1. Upload your files.
    2. Select pre-defined categories
    3. Verify that your file is parsable by clicking on the "Test Parse Files".
    4. Classify uploaded files into your pre-defined categories by clicking on the "Classify Documents" button.
    5. If you add/remove from current categories and want to recategorize your files, make sure you click the "Clear all documents" button at the bottom of the webpage.
    """)
    
    # Initialize handlers
    parser, db_handler, llm_handler, categorizer = init_handlers()
    
    # Create columns for category management
    st.subheader("Current Categories")
    
    for category in categorizer.get_categories():
        col1, col2 = st.columns([5, 1])
        with col1:
            st.text_input("Category", value=category, key=f"cat_{category}", disabled=True)
        with col2:
            if st.button("🗑️", key=f"del_{category}"):
                categorizer.remove_category(category)
                st.rerun()
    
    # File upload and processing section
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Test Parse Files"):
            for file in uploaded_files:
                try:
                    content = parser.parse(file)
                    st.success(f"Successfully parsed {file.name}")
                except Exception as e:
                    st.error(f"Error parsing {file.name}: {str(e)}")
        
        # Add a session state for controlling classification
        if 'stop_classification' not in st.session_state:
            st.session_state.stop_classification = False
            
        col1, col2 = st.columns([1, 1])
        with col1:
            classify_button = st.button("Classify Documents")
        with col2:
            if st.button("Force Quit Classification"):
                st.session_state.stop_classification = True
                st.warning("Classification stopped by user")
        
        if classify_button:
            st.session_state.stop_classification = False
            for file in uploaded_files:
                if st.session_state.stop_classification:
                    st.warning("Classification stopped by user")
                    break
                    
                try:
                    with st.spinner(f'Classifying {file.name}...'):
                        result = process_file(file, parser, db_handler, llm_handler, categorizer)
                        if result:
                            st.success(f"Classified {file.name} as {result.get('category', 'Unknown')}")
                except Exception as e:
                    st.error(f"Error categorizing {file.name}: {str(e)}")
    
    # Clear documents button at the bottom
    if st.button("Clear all documents"):
        db_handler.clear_all()
        st.success("All documents cleared from the database")

if __name__ == "__main__":
    main()