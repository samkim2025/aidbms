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
    
    # Add instructions in a clean format
    st.markdown("""
    ### Instructions
    1. **Upload your files**
       - Use the file uploader below to select your documents
       
    2. **Select pre-defined categories**
       - Add or remove categories in the 'Current Categories' section
       
    3. **Verify file parsing**
       - Click on "Test Parse Files" to ensure your files can be read correctly
       
    4. **Classify documents**
       - Click "Classify Documents" to categorize your files
       - Use "Force Quit" if you need to stop the classification process
       
    5. **Recategorizing files**
       - If you modify categories, click "Clear all documents" at the bottom
       - Then classify your documents again with the new categories
    """)
    
    # Add a visual separator
    st.markdown("---")
    
    # Initialize all session state variables
    if 'categories' not in st.session_state:
        st.session_state.categories = []
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = set()
    if 'file_categories' not in st.session_state:
        st.session_state.file_categories = {}

    # Add a sidebar for category management
    with st.sidebar:
        st.header("Category Management")
        
        # Add new category
        new_category = st.text_input("Add New Category")
        if st.button("Add Category"):
            if new_category and new_category not in st.session_state.categories:
                st.session_state.categories.append(new_category)
                # Trigger reclassification
                reclassify_all_documents()
                st.success(f"Added category: {new_category}")
            elif new_category in st.session_state.categories:
                st.warning("This category already exists!")
            else:
                st.warning("Please enter a category name!")
        
        # Show existing categories with delete buttons
        if st.session_state.categories:
            st.subheader("Current Categories")
            for idx, category in enumerate(st.session_state.categories):
                col1, col2 = st.columns([3, 1])
                with col1:
                    edited_category = st.text_input(
                        label=f"Category {idx+1}",
                        value=category,
                        key=f"edit_{idx}"
                    )
                    if edited_category != category:
                        st.session_state.categories[idx] = edited_category
                        # Trigger reclassification on edit
                        reclassify_all_documents()
                with col2:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        st.session_state.categories.pop(idx)
                        # Trigger reclassification on delete
                        reclassify_all_documents()
                        st.rerun()
        else:
            st.info("No categories added yet. Add your first category above!")

    # Main content area
    st.header("Upload and Classify Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload", 
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )
    
    # Test Parse Files button - moved before duplicate check
    if uploaded_files and st.button("Test Parse Files"):
        for file in uploaded_files:
            st.write(f"Testing parse for: {file.name}")
            content, error = read_file_content(file)
            if error:
                st.warning(f"Skipping {file.name}: {error}")
                continue
            
            st.write("First 200 characters of parsed content:")
            st.write(content[:200] if content else "No content extracted")
            st.write("---")
            file.seek(0)
    
    # Classify button
    if uploaded_files and st.button("🏷️ Classify Documents"):
        with st.spinner("Classifying documents..."):
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files_names:
                    content, error = read_file_content(file)
                    if error:
                        st.warning(f"Skipping {file.name}: {error}")
                        continue
                        
                    file.seek(0)
                    category, error = categorize_file(content, st.session_state.categories)
                    if error:
                        st.warning(f"Error categorizing {file.name}: {error}")
                        continue
                        
                    st.session_state.file_categories[file.name] = category
                    st.session_state.uploaded_files_names.add(file.name)
                    st.write(f"Classified {file.name} as: {category}")
                else:
                    st.info(f"Skipping {file.name} - already classified")
        st.success("Classification complete!")

    # Display files that are being processed
    if uploaded_files:
        st.subheader("Current Documents")
        for file in uploaded_files:
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                size = file.size
                st.write(f"{size/1000:.1f} KB")
            with col3:
                current_category = st.session_state.file_categories.get(file.name, "Uncategorized")
                categories_list = ['Uncategorized'] + st.session_state.categories
                
                try:
                    current_index = categories_list.index(current_category)
                except ValueError:
                    current_index = 0
                
                new_category = st.selectbox(
                    label=f"Category for {file.name}",
                    options=categories_list,
                    index=current_index,
                    key=f"cat_{file.name}"
                )
            
            with col4:
                if st.button("Save", key=f"save_{file.name}"):
                    st.session_state.file_categories[file.name] = new_category
                    st.session_state.uploaded_files_names.add(file.name)
                    st.success(f"Saved {file.name} in {new_category}")

    # Display categorized documents
    st.header("Categorized Documents")
    if 'file_categories' in st.session_state and st.session_state.file_categories:
        # Create tabs for each category including "Uncategorized"
        all_categories = ['Uncategorized'] + st.session_state.categories
        tabs = st.tabs(all_categories)
        
        # Organize files by category
        for tab, category in zip(tabs, all_categories):
            with tab:
                # Find all files in this category
                category_files = [
                    filename for filename, file_category 
                    in st.session_state.file_categories.items() 
                    if file_category == category
                ]
                
                if category_files:
                    for filename in category_files:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"📄 {filename}")
                        with col2:
                            if st.button("Remove", key=f"remove_{category}_{filename}"):
                                del st.session_state.file_categories[filename]
                                st.session_state.uploaded_files_names.remove(filename)
                                st.rerun()
                else:
                    st.write("No documents in this category")
    else:
        st.write("No documents have been categorized yet")

    # Add a clear all button
    if st.button("Clear All Documents"):
        if 'file_categories' in st.session_state:
            st.session_state.file_categories = {}
        st.session_state.uploaded_files_names = set()
        st.rerun()
                # Display documents here

def reclassify_all_documents():
    """Reclassify all documents when categories change"""
    if not st.session_state.uploaded_files_names:
        return

    with st.spinner("Reclassifying all documents..."):
        # Store old classifications for comparison
        old_classifications = st.session_state.file_categories.copy()
        
        # Reclassify each document
        for file_name in st.session_state.uploaded_files_names:
            try:
                content = read_file_content(file_name)
                if content and not isinstance(content, tuple):  # Check for valid content
                    new_category = categorize_file(content, st.session_state.categories)
                    st.session_state.file_categories[file_name] = new_category
                    
                    # Show changes in classification
                    if file_name in old_classifications and old_classifications[file_name] != new_category:
                        st.info(f"Reclassified: {file_name}\n"
                               f"From: {old_classifications[file_name]} → To: {new_category}")
            except Exception as e:
                st.error(f"Error reclassifying {file_name}: {str(e)}")
                # Keep old classification if error occurs
                if file_name in old_classifications:
                    st.session_state.file_categories[file_name] = old_classifications[file_name]
        
        st.success("Reclassification complete!")

if __name__ == "__main__":
    main()