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

# -------------------------------------------------------------------
# 1. INITIALIZATION
# -------------------------------------------------------------------
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

@timeout_handler(5)
def read_file_content(file):
    """Read content based on file type with timeout."""
    file_type = file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            content = PdfParser.parse(file)
            if not content:
                st.warning(f"Could not extract text from {file.name}. Possibly scanned or protected PDF.")
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

# -------------------------------------------------------------------
# 2. BASIC CATEGORIZATION HELPERS
# -------------------------------------------------------------------
def process_file(file, parser, db_handler, llm_handler, categorizer):
    """A sample function if you want to do single-file processing."""
    try:
        with st.spinner(f'Processing {file.name}...'):
            content = parser.parse(file)
            
            max_chunk = 4000
            if len(content) > max_chunk:
                # Break into chunks and combine
                chunks = [content[i:i+max_chunk] for i in range(0, len(content), max_chunk)]
                results = []
                for chunk in chunks:
                    result = llm_handler.categorize_content(chunk)
                    results.append(result)
                final_result = combine_results(results)
            else:
                final_result = llm_handler.categorize_content(content)
            
            db_handler.store_document(file.name, content, final_result)
            return final_result
            
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def combine_results(results):
    """Example aggregator if you break documents into multiple chunks."""
    categories = [r.get('category', 'Uncategorized') for r in results if r]
    return {'category': max(set(categories), key=categories.count)}

@timeout_handler(5)
def categorize_file(content, categories):
    """Categorize file content into one of the top-level categories."""
    if not categories or not content.strip():
        return "Uncategorized"
    
    prompt = f"""
    You are an expert document classifier with broad knowledge across multiple disciplines. 
    Categorize the following document into exactly ONE of these categories: {', '.join(categories)}

    Classification Guidelines:
    1. Content Analysis:
       - Consider main subject matter and themes
       - Look for specialized terminology
       - Evaluate overall focus, not just specific terms
    
    2. Decision Making:
       - Choose the category that best matches the primary topic
       - If content spans multiple categories, pick the most dominant theme
       - Use 'Uncategorized' only if no category clearly fits
    
    Document excerpt:
    ---
    {content[:3000]}
    ---

    Return ONLY the category name:
    """
    try:
        response = llm_handler.get_response(prompt)
        if isinstance(response, tuple):
            return "Uncategorized"
        response = response.strip()
        
        # Check case-insensitively
        for category in categories:
            if response.lower() == category.lower():
                return category
        return "Uncategorized"
    except Exception as e:
        print(f"Categorization error: {str(e)}")
        return "Uncategorized"

@timeout_handler(10)
def summarize_file_content(content, file_name):
    """Generate a short summary for the file content."""
    prompt = f"""
    Create a very brief summary of the following document in 1-2 short sentences.
    Focus on the main topic only. Start with:
    "{file_name} is a {file_name.split('.')[-1]} file about"

    Document excerpt:
    ---
    {content[:3000]}
    ---

    - Keep it under 200 characters
    - Use simple, direct language
    - Complete sentences only
    """
    return llm_handler.get_response(prompt)

# -------------------------------------------------------------------
# 3. MAIN STREAMLIT PAGES
# -------------------------------------------------------------------
def main():
    st.title("AI Document Management System")
    
    # Initialize session vars
    if 'categories' not in st.session_state:
        st.session_state.categories = []
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = set()
    if 'file_categories' not in st.session_state:
        st.session_state.file_categories = {}
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose functionality",
            ["Document Management", "Hierarchical Query Generator"],
            index=0
        )
        
        if page == "Document Management":
            st.markdown("---")
            st.header("Category Management")
            
            new_category = st.text_input("Add New Category")
            if st.button("Add Category"):
                if new_category and new_category not in st.session_state.categories:
                    st.session_state.categories.append(new_category)
                    reclassify_all_documents()
                    st.success(f"Added category: {new_category}")
                elif new_category in st.session_state.categories:
                    st.warning("This category already exists!")
                else:
                    st.warning("Please enter a category name!")
            
            # Show existing categories
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
                            reclassify_all_documents()
                    with col2:
                        if st.button("🗑️", key=f"delete_{idx}"):
                            st.session_state.categories.pop(idx)
                            reclassify_all_documents()
                            st.rerun()
            else:
                st.info("No categories added yet. Add your first category above!")
    
    # Page selection
    if page == "Document Management":
        document_management_page()
    else:
        hierarchical_query_page()

def document_management_page():
    """Handles the simpler Document Management UI."""
    st.markdown("""
    ### Instructions
    1. **Upload your files**.
    2. **Select pre-defined categories** on the sidebar.
    3. **Test Parse** (optional).
    4. **Classify Documents**.
    5. **Review** categories.
    6. **Summarize** if desired.
    """)
    st.markdown("---")
    
    if 'categories' not in st.session_state:
        st.session_state.categories = []
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = set()
    if 'file_categories' not in st.session_state:
        st.session_state.file_categories = {}

    st.header("Upload and Classify Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload", 
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )
    
    # Test Parse
    if uploaded_files and st.button("Test Parse Files"):
        for file in uploaded_files:
            st.write(f"Testing parse for: {file.name}")
            content, error = read_file_content(file)
            if error:
                st.warning(f"Skipping {file.name}: {error}")
                continue
            st.write("First 200 characters:")
            st.write(content[:200] if content else "No content extracted")
            st.write("---")
            file.seek(0)
    
    # Classify
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
        st.success("Done!")

    # Show documents in a table
    if uploaded_files:
        st.subheader("Current Documents")
        for file in uploaded_files:
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                size_kb = file.size / 1024
                st.write(f"{size_kb:.1f} KB")
            with col3:
                current_category = st.session_state.file_categories.get(file.name, "Uncategorized")
                all_cats = ['Uncategorized'] + st.session_state.categories
                try:
                    current_index = all_cats.index(current_category)
                except ValueError:
                    current_index = 0
                new_cat = st.selectbox(
                    f"Category for {file.name}",
                    options=all_cats,
                    index=current_index,
                    key=f"cat_{file.name}"
                )
            with col4:
                if st.button("Save", key=f"save_{file.name}"):
                    st.session_state.file_categories[file.name] = new_cat
                    st.session_state.uploaded_files_names.add(file.name)
                    st.success(f"Saved {file.name} in {new_cat}")

    st.header("Categorized Documents")
    if st.session_state.file_categories:
        all_cat_tabs = ['Uncategorized'] + st.session_state.categories
        tabs = st.tabs(all_cat_tabs)
        
        for tab, category in zip(tabs, all_cat_tabs):
            with tab:
                cat_files = [f for f, cat in st.session_state.file_categories.items() if cat == category]
                if cat_files:
                    for f in cat_files:
                        col_a, col_b = st.columns([4,1])
                        with col_a:
                            st.write(f"📄 {f}")
                        with col_b:
                            if st.button("Remove", key=f"remove_{category}_{f}"):
                                del st.session_state.file_categories[f]
                                st.session_state.uploaded_files_names.remove(f)
                                st.rerun()
                else:
                    st.write("No documents in this category")
    else:
        st.write("No documents have been categorized yet")

    # Clear all
    if st.button("Clear All Documents"):
        st.session_state.file_categories = {}
        st.session_state.uploaded_files_names = set()
        st.rerun()

    # Summaries
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
                            summary = summary_tuple[0]  # if it returns (text, None)
                            st.markdown(f"**Summary**: {summary}")
                            st.write("---")
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                    finally:
                        file.seek(0)

def reclassify_all_documents():
    """If categories are changed or removed, re-run classification on previously uploaded docs."""
    if not st.session_state.uploaded_files_names:
        return
    with st.spinner("Reclassifying all documents..."):
        old_cats = st.session_state.file_categories.copy()
        for fname in st.session_state.uploaded_files_names:
            try:
                content = read_file_content(fname)
                if content and not isinstance(content, tuple):
                    new_cat, error = categorize_file(content, st.session_state.categories)
                    if not error:
                        st.session_state.file_categories[fname] = new_cat
                        if fname in old_cats and old_cats[fname] != new_cat:
                            st.info(f"Reclassified {fname}\nFrom: {old_cats[fname]} → To: {new_cat}")
            except Exception as e:
                st.error(f"Error reclassifying {fname}: {str(e)}")
                if fname in old_cats:
                    st.session_state.file_categories[fname] = old_cats[fname]
        st.success("Reclassification complete!")

# -------------------------------------------------------------------
# 4. HIERARCHICAL CLASSIFICATION PAGE
# -------------------------------------------------------------------
def hierarchical_query_page():
    """Manage hierarchical classification logic."""
    st.title("Hierarchical Query Path Generator")
    
    if 'hierarchy_depth' not in st.session_state:
        st.session_state.hierarchy_depth = None
    if 'hierarchy' not in st.session_state:
        st.session_state.hierarchy = {}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    left_col, right_col = st.columns([2, 3])
    
    with left_col:
        st.markdown("""
        ### 📊 Hierarchy Setup
        1. Set the depth
        2. Define categories
        3. Upload documents
        4. Generate hierarchy
        """)
        
        st.markdown("#### Step 1: Set Hierarchy Depth")
        depth_container = st.container()
        with depth_container:
            if st.session_state.hierarchy_depth is None:
                depth = st.select_slider(
                    "Select hierarchy depth",
                    options=[2, 3, 4],
                    value=2
                )
                if st.button("✨ Initialize Hierarchy"):
                    st.session_state.hierarchy_depth = depth
                    st.session_state.hierarchy = {}
                    st.rerun()
            else:
                st.info(f"Current hierarchy depth: {st.session_state.hierarchy_depth} levels")
                if st.button("🔄 Reset Hierarchy"):
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
            st.success(f"Uploaded {len(uploaded_files)} documents.")
        
        # Step 4: Generate
        st.markdown("#### Step 4: Generate Hierarchy")
        if st.session_state.hierarchy and st.session_state.uploaded_files:
            if st.button("🚀 Generate Hierarchy"):
                with st.spinner("Processing files..."):
                    try:
                        results = process_files_for_hierarchy(
                            st.session_state.uploaded_files,
                            st.session_state.hierarchy
                        )
                        st.session_state.hierarchy_results = results
                        st.success("Hierarchy generated!")
                        
                        nodes, edges, config = visualize_hierarchy_results(results)
                        st.markdown("### Hierarchy Visualization")
                        agraph(nodes=nodes, edges=edges, config=config)
                        
                        st.markdown("### Export Results")
                        csv = export_hierarchy_results(results)
                        st.download_button(
                            "Download CSV",
                            data=csv,
                            file_name="hierarchy_results.csv",
                            mime="text/csv"
                        )
                        
                        st.markdown("### Detailed Results")
                        for doc_name, doc_info in results.items():
                            with st.expander(f"📄 {doc_name}"):
                                st.write(f"**Category Path:** {doc_info['path']}")
                                st.write(f"**Preview:** {doc_info['content']}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("Complete steps 1-3 first")
    
    with right_col:
        if st.session_state.hierarchy_depth is not None:
            st.markdown("#### Step 2: Define Categories")
            level_tabs = st.tabs([f"Level {i+1}" for i in range(st.session_state.hierarchy_depth)])
            
            def add_category(parent_path="", level=1):
                with level_tabs[level-1]:
                    if parent_path:
                        st.markdown(f"*Current path: {parent_path}*")
                    
                    col1, col2 = st.columns([3,1])
                    with col1:
                        new_cat = st.text_input(
                            "Category name",
                            key=f"new_cat_{parent_path}_{level}"
                        )
                    with col2:
                        add_btn = st.button("➕ Add", key=f"add_{parent_path}_{level}")
                    
                    if add_btn and new_cat:
                        cur_level = st.session_state.hierarchy
                        pparts = parent_path.split("/") if parent_path else []
                        
                        for p in pparts:
                            if p:
                                cur_level = cur_level[p]
                        
                        if isinstance(cur_level, dict):
                            if new_cat not in cur_level:
                                if level < st.session_state.hierarchy_depth:
                                    cur_level[new_cat] = {}
                                else:
                                    cur_level[new_cat] = {}
                                st.success(f"Added: {new_cat}")
                                st.rerun()
                    
                    if st.session_state.hierarchy:
                        st.markdown("##### Current Categories:")
                        cur_level = st.session_state.hierarchy
                        for p in parent_path.split("/"):
                            if p:
                                cur_level = cur_level[p]
                        
                        if isinstance(cur_level, dict):
                            for cat in cur_level:
                                col_a, col_b = st.columns([4,1])
                                with col_a:
                                    st.markdown(f"🔹 {cat}")
                                with col_b:
                                    if st.button("🗑️", key=f"del_{parent_path}_{cat}"):
                                        del cur_level[cat]
                                        st.rerun()
                                
                                # Recursively show deeper subcats
                                if level < st.session_state.hierarchy_depth:
                                    new_p = f"{parent_path}/{cat}" if parent_path else cat
                                    add_category(new_p, level+1)

            add_category()
            st.markdown("#### Complete Hierarchy Preview")
            if st.session_state.hierarchy:
                def display_hierarchy(data, lvl=0):
                    out = ""
                    indent = "    " * lvl
                    for cat, subs in data.items():
                        out += f"{indent}📁 {cat}\n"
                        if isinstance(subs, dict):
                            out += display_hierarchy(subs, lvl+1)
                    return out
                
                st.code(display_hierarchy(st.session_state.hierarchy), language="plaintext")
            else:
                st.info("Add categories to see the preview.")

# -------------------------------------------------------------------
# 5. ADVANCED HIERARCHICAL LOGIC
# -------------------------------------------------------------------
def score_category(content, category, parent_path=""):
    """
    Prompt the LLM to get a relevance score (0 to 1) for how well the content
    matches the given 'category', using the parent's path for extra context.
    """
    # If there's a parent path, we incorporate it:
    # e.g. "Cars/European" plus "Mercedes"
    if parent_path:
        path_string = f"Parent categories so far: {parent_path}"
    else:
        path_string = "No parent (root level)."
    
    prompt = f"""
    You are classifying a document within a hierarchical taxonomy. 
    {path_string}

    Rate how well this document fits the next sub-category: '{category}'.

    The rating scale is between 0 and 1:
    - 1 means a perfect match
    - 0 means no match

    Return only a numeric score (no text, no explanation).

    Document excerpt (up to 1000 chars):
    {content[:1000]}
    """
    try:
        response = llm_handler.get_response(prompt)
        response = response.strip().split()[0]
        response = ''.join(c for c in response if c.isdigit() or c == '.')
        if response:
            return float(response)
        return 0.0
    except Exception as e:
        print(f"Error scoring category '{category}': {str(e)}")
        return 0.0

def find_best_category_path(content, hierarchy, current_path="", threshold=0.3):
    """
    Recursively find the best scoring path in the hierarchy.
      - We include the parent's path in the prompt.
      - If the child ties or beats the parent, we descend.

    Returns (best_path, best_score).
    """
    best_local_path = "Uncategorized"
    best_local_score = 0.0
    
    for category, subcats in hierarchy.items():
        # 1. Score the current category, given the parent's path
        score = score_category(content, category, parent_path=current_path)
        st.write(f"DEBUG: Path '{current_path or '(root)'}', checking category '{category}', score={score:.4f}")
        
        new_path = f"{current_path}/{category}" if current_path else category
        candidate_path = new_path
        candidate_score = score
        
        # 2. If we pass threshold and have subcategories, go deeper
        if (isinstance(subcats, dict) and score >= threshold and len(subcats) > 0):
            deeper_path, deeper_score = find_best_category_path(content, subcats, new_path, threshold)
            # If child is >= the parent's score, pick the deeper path
            if deeper_score >= candidate_score:
                candidate_path = deeper_path
                candidate_score = deeper_score
        
        # 3. Compare with best so far at this level
        if candidate_score > best_local_score:
            best_local_score = candidate_score
            best_local_path = candidate_path
    
    return best_local_path, best_local_score

def find_category_path(content, hierarchy, threshold=0.3):
    """
    A wrapper to call the recursive function and return just the best path 
    or 'Uncategorized' if it's too low.
    """
    best_path, best_score = find_best_category_path(content, hierarchy, "", threshold)
    st.write(f"DEBUG: Best path = '{best_path}', best_score={best_score:.4f}, threshold={threshold}")
    if best_score < 0.05:
        return "Uncategorized"
    return best_path

def process_files_for_hierarchy(files, hierarchy):
    """Classify each file into the multi-level hierarchy."""
    processed_results = {}
    
    for file in files:
        try:
            content, error = read_file_content(file)
            if error:
                continue
            st.write(f"---\n**Processing file:** {file.name}")
            
            path = find_category_path(content, hierarchy, threshold=0.3)
            
            processed_results[file.name] = {
                'path': path,
                'content': content[:200] + "..."
            }
            st.write(f"DEBUG: Final chosen path for '{file.name}' = {path}")
            st.write("---")
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    return processed_results

def visualize_hierarchy_results(hierarchy_results):
    """Create an interactive hierarchy graph with streamlit_agraph."""
    nodes = []
    edges = []
    
    def add_category_nodes(hier_dict, parent_id=None, level=0):
        for cat, subs in hier_dict.items():
            node_id = f"cat_{level}_{cat}"
            nodes.append(Node(
                id=node_id,
                label=cat,
                size=20,
                color="#ff9999",
                shape="dot"
            ))
            if parent_id:
                edges.append(Edge(source=parent_id, target=node_id))
            if isinstance(subs, dict):
                add_category_nodes(subs, node_id, level+1)
    
    def add_doc_nodes(results):
        for doc_name, doc_info in results.items():
            doc_id = f"doc_{doc_name}"
            nodes.append(Node(
                id=doc_id,
                label=doc_name,
                size=15,
                color="#99ff99",
                shape="square"
            ))
            
            if doc_info['path'] != "Uncategorized":
                parts = doc_info['path'].split('/')
                last_cat = parts[-1]
                cat_id = f"cat_{len(parts)-1}_{last_cat}"
                edges.append(Edge(source=cat_id, target=doc_id))
            else:
                # If uncategorized, create or link to a special node
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
    
    add_category_nodes(st.session_state.hierarchy)
    add_doc_nodes(hierarchy_results)
    
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
    """Generate a CSV from the hierarchical results."""
    import pandas as pd
    data = []
    for doc_name, doc_info in hierarchy_results.items():
        data.append({
            'Document': doc_name,
            'Category Path': doc_info['path'],
            'Preview': doc_info['content']
        })
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# -------------------------------------------------------------------
# 6. RUN STREAMLIT
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
