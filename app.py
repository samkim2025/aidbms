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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict
from streamlit_agraph import agraph, Node, Edge, Config

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

@timeout_handler(10)
def summarize_file_content(content, file_name):
    """Generate a summary for file content with timeout"""
    prompt = f"""
    Create a very brief summary of the following document in 1-2 short sentences. 
    Be concise and focus on the main topic only.
    Start with: "{file_name} is a {file_name.split('.')[-1]} file about"
    
    Document content:
    ---
    {content[:3000]}
    ---
    
    Guidelines:
    - Keep the summary under 200 characters
    - Focus on the core topic only
    - Use simple, direct language
    - Complete sentences only (no mid-sentence cutoffs)
    """
    
    return llm_handler.get_response(prompt)

def main():
    st.title("AI Document Management System")
    
    # Initialize all session state variables
    if 'categories' not in st.session_state:
        st.session_state.categories = []
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = set()
    if 'file_categories' not in st.session_state:
        st.session_state.file_categories = {}
    
    # Move navigation to sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose functionality",
            ["Document Management", "Hierarchical Query Generator"],
            index=0
        )
        
        # Only show Category Management if we're in Document Management page
        if page == "Document Management":
            st.markdown("---")  # Add separator
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
    
    # Display the selected page
    if page == "Document Management":
        document_management_page()
    else:
        hierarchical_query_page()

def document_management_page():
    """Contains all the existing document management functionality"""
    # Add instructions in a clean format
    st.markdown("""
    ### Instructions
    1. **Upload your files**: Use the file uploader below to select your documents.
       
    2. **Select pre-defined categories**: Add or remove categories in the 'Current Categories' section.
       
    3. **Verify file parsing**: Click on "Test Parse Files" to ensure your files can be read correctly.
       
    4. **Classify documents**: Click "Classify Documents" to categorize your files.
       
    5. **Recategorizing files**: If you modify categories, click "Clear all documents" at the bottom. Then classify your documents again with the new categories.
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

    # Main content area
    st.header("Upload and Classify Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload", 
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )
    
    # Test Parse Files button
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

    # Add Summarize Files section
    st.header("Summarize Files")
    if uploaded_files:
        for file in uploaded_files:
            st.write(f"📄 {file.name}")
            if st.button("Summarize", key=f"summarize_{file.name}"):
                with st.spinner(f"Summarizing {file.name}..."):
                    try:
                        content, error = read_file_content(file)
                        if error:
                            st.warning(f"Skipping {file.name}: {error}")
                            continue
                        
                        summary_tuple = summarize_file_content(content, file.name)
                        if isinstance(summary_tuple, tuple):
                            summary = summary_tuple[0]  # Get the actual summary text
                            st.write("Summary:")
                            st.markdown(f"_{summary}_")
                            st.write("---")
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                    finally:
                        file.seek(0)  # Reset file pointer

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

def hierarchical_query_page():
    """Page for hierarchical query path generator"""
    st.title("Hierarchical Query Path Generator")
    
    # Initialize session state
    if 'hierarchy_depth' not in st.session_state:
        st.session_state.hierarchy_depth = None
    if 'hierarchy' not in st.session_state:
        st.session_state.hierarchy = {}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Create two columns for main layout
    left_col, right_col = st.columns([2, 3])
    
    with left_col:
        st.markdown("""
        ### 📊 Hierarchy Setup
        Follow these steps to create your category hierarchy:
        1. Set the depth of your hierarchy
        2. Define categories at each level
        3. Upload your documents
        4. Generate the hierarchy
        """)
        
        # Step 1: Set hierarchy depth with visual feedback
        st.markdown("#### Step 1: Set Hierarchy Depth")
        depth_container = st.container()
        with depth_container:
            if st.session_state.hierarchy_depth is None:
                depth = st.select_slider(
                    "Select hierarchy depth",
                    options=[2, 3, 4],
                    value=2,
                    help="Choose how many levels deep your category hierarchy should be"
                )
                if st.button("✨ Initialize Hierarchy", type="primary"):
                    st.session_state.hierarchy_depth = depth
                    st.session_state.hierarchy = {}
                    st.rerun()
            else:
                st.info(f"🌳 Current hierarchy depth: {st.session_state.hierarchy_depth} levels")
                if st.button("🔄 Reset Hierarchy", type="secondary"):
                    st.session_state.hierarchy_depth = None
                    st.session_state.hierarchy = {}
                    st.rerun()
        
        # Step 3: File Upload
        st.markdown("#### Step 3: Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose documents to classify",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx']
        )
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"📁 {len(uploaded_files)} documents uploaded")
        
        # Step 4: Generate Hierarchy
        st.markdown("#### Step 4: Generate Hierarchy")
        if st.session_state.hierarchy and st.session_state.uploaded_files:
            if st.button("🚀 Generate Hierarchy", type="primary"):
                with st.spinner("Processing files and generating hierarchy..."):
                    try:
                        # Process files
                        results = process_files_for_hierarchy(
                            st.session_state.uploaded_files,
                            st.session_state.hierarchy
                        )
                        
                        # Store results in session state
                        st.session_state.hierarchy_results = results
                        
                        # Show success message
                        st.success("✅ Hierarchy generated successfully!")
                        
                        # Create visualization
                        nodes, edges, config = visualize_hierarchy_results(results)
                        
                        # Display visualization
                        st.markdown("### Hierarchy Visualization")
                        agraph(nodes=nodes, edges=edges, config=config)
                        
                        # Export option
                        st.markdown("### Export Results")
                        csv = export_hierarchy_results(results)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="hierarchy_results.csv",
                            mime="text/csv"
                        )
                        
                        # Display results in expandable sections
                        st.markdown("### Detailed Results")
                        for doc_name, doc_info in results.items():
                            with st.expander(f"📄 {doc_name}"):
                                st.write(f"**Category Path:** {doc_info['path']}")
                                st.write(f"**Preview:** {doc_info['content']}")
                    except Exception as e:
                        st.error(f"Error generating hierarchy: {str(e)}")
        else:
            st.warning("⚠️ Please complete steps 1-3 first")
    
    with right_col:
        # Step 2: Category Definition
        if st.session_state.hierarchy_depth is not None:
            st.markdown("#### Step 2: Define Categories")
            
            # Create tabs for each level
            level_tabs = st.tabs([f"Level {i+1}" for i in range(st.session_state.hierarchy_depth)])
            
            def add_category(parent_path="", level=1):
                with level_tabs[level-1]:
                    # Show current path
                    if parent_path:
                        st.markdown(f"*Current path: {parent_path}*")
                    
                    # Add new category input
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        new_cat = st.text_input(
                            "Category name",
                            key=f"new_cat_{parent_path}_{level}",
                            placeholder="Enter category name..."
                        )
                    with col2:
                        add_button = st.button(
                            "➕ Add",
                            key=f"add_{parent_path}_{level}",
                            type="primary"
                        )
                    
                    if add_button and new_cat:
                        current_level = st.session_state.hierarchy
                        path_parts = parent_path.split("/") if parent_path else []
                        
                        for part in path_parts:
                            if part:
                                current_level = current_level[part]
                        
                        if isinstance(current_level, dict):
                            if new_cat not in current_level:
                                # If not the last level, init as dict
                                # If it's the last level, can keep as dict or None 
                                # for leaf. We'll keep it as dict for consistency
                                if level < st.session_state.hierarchy_depth:
                                    current_level[new_cat] = {}
                                else:
                                    current_level[new_cat] = {}
                                st.success(f"✅ Added: {new_cat}")
                                st.rerun()
                    
                    # Display existing categories
                    if st.session_state.hierarchy:
                        st.markdown("##### Current Categories:")
                        current_level = st.session_state.hierarchy
                        for part in parent_path.split("/"):
                            if part:
                                current_level = current_level[part]
                        
                        if isinstance(current_level, dict):
                            for category in current_level:
                                cat_col1, cat_col2 = st.columns([4, 1])
                                with cat_col1:
                                    st.markdown(f"🔹 {category}")
                                with cat_col2:
                                    if st.button("🗑️", key=f"del_{parent_path}_{category}"):
                                        del current_level[category]
                                        st.rerun()
                                
                                # Recursively add subcategories
                                if level < st.session_state.hierarchy_depth:
                                    new_path = f"{parent_path}/{category}" if parent_path else category
                                    add_category(new_path, level + 1)
            
            # Start building hierarchy from root
            add_category()
            
            # Display current complete hierarchy
            st.markdown("#### Complete Hierarchy Preview")
            if st.session_state.hierarchy:
                def display_hierarchy(data, level=0):
                    result = ""
                    indent = "    " * level
                    for category, subcategories in data.items():
                        result += f"{indent}📁 {category}\n"
                        if isinstance(subcategories, dict):
                            result += display_hierarchy(subcategories, level + 1)
                    return result
                
                st.code(display_hierarchy(st.session_state.hierarchy), language="plaintext")
            else:
                st.info("Start adding categories to see the hierarchy preview")

def process_files_for_hierarchy(files, hierarchy):
    """Process files and classify them according to the defined hierarchy"""
    processed_results = {}
    
    for file in files:
        try:
            # Read file content
            content, error = read_file_content(file)
            if error:
                continue
            
            # Find the best matching path in hierarchy
            best_path = find_category_path(content, hierarchy)
            
            # Store result
            processed_results[file.name] = {
                'path': best_path,
                'content': content[:200] + "..."  # Store preview
            }
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    return processed_results

def score_category(content, category):
    """
    Prompt the LLM to get a relevance score (0 to 1) 
    for how well the content matches the given category.
    """
    prompt = f"""
    Rate how well this document matches the category '{category}'.

    Document excerpt:
    {content[:1000]}...

    Instructions:
    - Respond with ONLY a number between 0 and 1
    - Use 1 for perfect match
    - Use 0 for no match
    - Do not include any explanation
    - Just the number, nothing else

    Rating:
    """
    try:
        response = llm_handler.get_response(prompt)
        response = response.strip().split()[0]  # Take first word only
        # Remove any non-numeric characters except decimal point
        response = ''.join(c for c in response if c.isdigit() or c == '.')
        if response:
            return float(response)
        else:
            return 0.0
    except Exception as e:
        print(f"Error scoring category {category}: {str(e)}")
        return 0.0

def find_category_path(content, hierarchy, current_path="", threshold=0.6):
    """
    Recursively find the best matching category path for the content using a threshold.
    If a leaf node is relevant (score > threshold), place the file at that leaf. 
    Otherwise, keep it at the best-matching parent.

    Returns:
       best_path (str): The path in the hierarchy that best matches the content.
    """
    best_local_score = 0.0
    best_local_path = "Uncategorized"
    
    # We compare the highest score among siblings
    for category, subcategories in hierarchy.items():
        # 1. Score current category
        category_score = score_category(content, category)

        # Build path
        new_path = f"{current_path}/{category}" if current_path else category
        
        if category_score > best_local_score:
            best_local_score = category_score
            best_local_path = new_path
        
        # 2. If category has subcategories (dict) and we exceed threshold, 
        #    then attempt to find a deeper match
        if isinstance(subcategories, dict) and category_score >= threshold:
            deeper_path = find_category_path(content, subcategories, new_path, threshold)
            # If deeper_path is not "Uncategorized", let's get its last score from that path
            if deeper_path != "Uncategorized":
                # We'll compare the parent's "best_local_score" with the child's final score
                # The child's final score is the last category in deeper_path
                child_category = deeper_path.split('/')[-1]
                child_score = score_category(content, child_category)
                
                if child_score > best_local_score:
                    best_local_score = child_score
                    best_local_path = deeper_path
    
    # If we never found a category above 0, or best_local_score is extremely low,
    # we can return "Uncategorized". Adjust as needed.
    # But if we found some match, return that path.
    if best_local_score < 0.01:  # If everything is near 0, call it "Uncategorized"
        return "Uncategorized"
    
    return best_local_path

def visualize_hierarchy_results(hierarchy_results):
    """Create an interactive visualization of the hierarchy results"""
    nodes = []
    edges = []
    
    # Create nodes for each category level
    def add_category_nodes(hierarchy, parent_id=None, level=0):
        for category, subcategories in hierarchy.items():
            current_id = f"cat_{level}_{category}"
            nodes.append(Node(
                id=current_id,
                label=category,
                size=20,
                color="#ff9999",  # Red-ish for categories
                shape="dot"
            ))
            
            if parent_id:
                edges.append(Edge(source=parent_id, target=current_id))
            
            if isinstance(subcategories, dict):
                add_category_nodes(subcategories, current_id, level + 1)
    
    # Add document nodes
    def add_document_nodes(results):
        for doc_name, doc_info in results.items():
            doc_id = f"doc_{doc_name}"
            nodes.append(Node(
                id=doc_id,
                label=doc_name,
                size=15,
                color="#99ff99",  # Green-ish for documents
                shape="square"
            ))
            
            if doc_info['path'] != "Uncategorized":
                # Connect to its category
                path_parts = doc_info['path'].split('/')
                # The last part is the actual category name
                final_cat = path_parts[-1]
                target_id = f"cat_{len(path_parts)-1}_{final_cat}"
                edges.append(Edge(source=target_id, target=doc_id))
            else:
                # If it's Uncategorized, we can create a separate node or skip
                unc_id = "cat_uncategorized"
                if not any(n.id == unc_id for n in nodes):
                    nodes.append(Node(
                        id=unc_id,
                        label="Uncategorized",
                        size=20,
                        color="#fab1ce",
                        shape="dot"
                    ))
                edges.append(Edge(source=unc_id, target=doc_id))
    
    # Create the visualization
    add_category_nodes(st.session_state.hierarchy)
    add_document_nodes(hierarchy_results)
    
    # Configure the graph
    config = Config(
        width=800,
        height=600,
        directed=True,
        physics=True,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True
    )
    
    return nodes, edges, config

def export_hierarchy_results(hierarchy_results):
    """Export hierarchy results to CSV"""
    import pandas as pd
    import io
    
    # Create DataFrame
    data = []
    for doc_name, doc_info in hierarchy_results.items():
        data.append({
            'Document': doc_name,
            'Category Path': doc_info['path'],
            'Preview': doc_info['content']
        })
    
    df = pd.DataFrame(data)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    return csv

if __name__ == "__main__":
    main()
