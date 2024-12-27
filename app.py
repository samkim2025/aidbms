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
import numpy as np
from collections import defaultdict
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="Document Navigator", layout="wide")

###############################################################################
# 1. INITIAL SETUP: Caching, Timeout Decorators, Handler Init
###############################################################################

@st.cache_resource
def init_handlers():
    parser = DocumentParser()
    db_handler = DatabaseHandler()
    llm_handler = LLMHandler()
    categorizer = AICategorizer(llm_handler)
    return parser, db_handler, llm_handler, categorizer

parser, db_handler, llm_handler, categorizer = init_handlers()

def timeout_handler(timeout_duration=5):
    """Decorator to handle timeouts for functions."""
    def decorator(func):
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
        return wraps(func)(wrapper)
    return decorator

@timeout_handler(5)
def read_file_content(file):
    """Read the file content (pdf, txt, docx) with a 5-second timeout."""
    file_type = file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            content = PdfParser.parse(file)
            if not content:
                st.warning(f"Could not extract text from {file.name} (maybe scanned/protected).")
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

###############################################################################
# 2. BASIC TOP-LEVEL CATEGORIZATION (FOR 'DOCUMENT MANAGEMENT' PAGE)
###############################################################################

@timeout_handler(5)
def categorize_file(content, categories):
    """
    A simpler function that classifies a document into ONE of the top-level 
    categories. This is for the Document Management page, not the deeper 
    hierarchical classification.
    """
    if not categories or not content.strip():
        return "Uncategorized"
    
    prompt = f"""
    You are an expert classifier. Classify this text into exactly ONE of these 
    top-level categories: {', '.join(categories)}.

    If it doesn't match well, return 'Uncategorized'.

    Text snippet (up to 3000 chars):
    ---
    {content[:3000]}
    ---

    Return ONLY the single category name:
    """
    try:
        response = llm_handler.get_response(prompt)
        if isinstance(response, tuple):
            return "Uncategorized"
        
        response = response.strip()
        for cat in categories:
            if response.lower() == cat.lower():
                return cat
        return "Uncategorized"
    except Exception as e:
        print(f"Top-level categorization error: {str(e)}")
        return "Uncategorized"

def reclassify_all_documents():
    """
    If user modifies or removes top-level categories, re-run 
    the classification on already uploaded docs.
    """
    if not st.session_state.get('uploaded_files_names'):
        return
    
    with st.spinner("Reclassifying..."):
        old_cats = st.session_state.file_categories.copy()
        for fname in st.session_state.uploaded_files_names:
            try:
                content = read_file_content(fname)
                if content and not isinstance(content, tuple):
                    new_cat, err = categorize_file(content, st.session_state.categories)
                    if not err:
                        st.session_state.file_categories[fname] = new_cat
                        
                        # If changed
                        if fname in old_cats and old_cats[fname] != new_cat:
                            st.info(f"Reclassified {fname}: {old_cats[fname]} → {new_cat}")
            except Exception as e:
                st.error(f"Error reclassifying {fname}: {str(e)}")
                if fname in old_cats:
                    st.session_state.file_categories[fname] = old_cats[fname]
        st.success("Reclassification complete!")

###############################################################################
# 3. SUMMARIZATION HELPERS
###############################################################################

@timeout_handler(10)
def summarize_file_content(content, file_name):
    """
    Generate a short summary for the file content. 
    Returns a text summary or (summary, None).
    """
    prompt = f"""
    Create a concise summary (1-2 sentences, under 200 characters) 
    for the following document. Start with:
    "{file_name} is a {file_name.split('.')[-1]} file about..."

    Document excerpt:
    ---
    {content[:3000]}
    ---

    Return ONLY the summary text:
    """
    return llm_handler.get_response(prompt)

###############################################################################
# 4. STREAMLIT MAIN AND PAGE NAVIGATION
###############################################################################

def main():
    st.title("AI Document Management System")

    # Initialize session
    if 'categories' not in st.session_state:
        st.session_state.categories = []
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = set()
    if 'file_categories' not in st.session_state:
        st.session_state.file_categories = {}

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose functionality",
            ["Document Management", "Hierarchical Query Generator"],
            index=0
        )
        
        # If user is on Document Management, show category mgmt
        if page == "Document Management":
            st.markdown("---")
            st.header("Category Management")
            new_cat = st.text_input("Add New Category")
            if st.button("Add Category"):
                if new_cat and new_cat not in st.session_state.categories:
                    st.session_state.categories.append(new_cat)
                    reclassify_all_documents()
                    st.success(f"Added category: {new_cat}")
                elif new_cat in st.session_state.categories:
                    st.warning("That category already exists!")
                else:
                    st.warning("Enter a category name!")

            if st.session_state.categories:
                st.subheader("Current Categories")
                for idx, cat in enumerate(st.session_state.categories):
                    c1, c2 = st.columns([3,1])
                    with c1:
                        edited = st.text_input(f"Category {idx+1}", value=cat, key=f"edit_{idx}")
                        if edited != cat:
                            st.session_state.categories[idx] = edited
                            reclassify_all_documents()
                    with c2:
                        if st.button("🗑️", key=f"delete_{idx}"):
                            st.session_state.categories.pop(idx)
                            reclassify_all_documents()
                            st.rerun()
            else:
                st.info("No top-level categories yet. Add one above!")
    
    # Pages
    if page == "Document Management":
        document_management_page()
    else:
        hierarchical_query_page()

###############################################################################
# 5. DOCUMENT MANAGEMENT PAGE
###############################################################################

def document_management_page():
    """Lets user upload, parse, top-level classify, and summarize docs."""
    st.markdown("""
    ### Basic Instructions
    1. **Upload** your files.
    2. **Test Parse** if you want to see the text excerpt.
    3. **Classify** into top-level categories on the sidebar.
    4. **Save** or **Remove** them from categories.
    5. Optionally **Summarize**.
    """)
    st.markdown("---")

    # Upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=['pdf','txt','docx']
    )

    # Test parse
    if uploaded_files and st.button("Test Parse Files"):
        for file in uploaded_files:
            st.write(f"Testing parse for {file.name}")
            content, err = read_file_content(file)
            if err:
                st.warning(f"Skipping {file.name}: {err}")
                continue
            st.write("First 200 chars:")
            st.write(content[:200])
            file.seek(0)
            st.write("---")

    # Classify
    if uploaded_files and st.button("Classify Documents"):
        with st.spinner("Classifying..."):
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files_names:
                    content, err = read_file_content(file)
                    if err:
                        st.warning(f"Skipping {file.name}: {err}")
                        continue
                    cat, cat_err = categorize_file(content, st.session_state.categories)
                    if cat_err:
                        st.warning(f"Error: {cat_err}")
                        continue
                    st.session_state.file_categories[file.name] = cat
                    st.session_state.uploaded_files_names.add(file.name)
                    st.write(f"{file.name} → {cat}")
        st.success("Done!")

    # Display docs
    if uploaded_files:
        st.subheader("Current Documents")
        for file in uploaded_files:
            c1, c2, c3, c4 = st.columns([2,1,2,1])
            with c1:
                st.write(f"📄 {file.name}")
            with c2:
                st.write(f"{file.size/1024:.1f} KB")
            with c3:
                current = st.session_state.file_categories.get(file.name, "Uncategorized")
                cat_opts = ["Uncategorized"] + st.session_state.categories
                try:
                    idx = cat_opts.index(current)
                except ValueError:
                    idx = 0
                new_cat = st.selectbox(
                    f"Category for {file.name}",
                    cat_opts,
                    index=idx,
                    key=f"cat_{file.name}"
                )
            with c4:
                if st.button("Save", key=f"save_{file.name}"):
                    st.session_state.file_categories[file.name] = new_cat
                    st.session_state.uploaded_files_names.add(file.name)
                    st.success(f"Saved {file.name} in {new_cat}")
    
    # Show categorized docs
    st.header("Categorized Documents")
    if st.session_state.file_categories:
        all_cats = ["Uncategorized"] + st.session_state.categories
        tabs = st.tabs(all_cats)
        for tab, c in zip(tabs, all_cats):
            with tab:
                docs_in_cat = [fname for fname, cat in st.session_state.file_categories.items() if cat == c]
                if docs_in_cat:
                    for d in docs_in_cat:
                        col_a, col_b = st.columns([4,1])
                        with col_a:
                            st.write(f"📄 {d}")
                        with col_b:
                            if st.button("Remove", key=f"remove_{c}_{d}"):
                                del st.session_state.file_categories[d]
                                st.session_state.uploaded_files_names.remove(d)
                                st.rerun()
                else:
                    st.write("No documents in this category.")
    else:
        st.write("No documents have been classified yet.")

    # Clear all
    if st.button("Clear All Documents"):
        st.session_state.file_categories.clear()
        st.session_state.uploaded_files_names.clear()
        st.rerun()

    # Summaries
    st.header("Summarize Files")
    if uploaded_files:
        for file in uploaded_files:
            st.write(f"📄 {file.name}")
            if st.button("Summarize", key=f"summarize_{file.name}"):
                with st.spinner(f"Summarizing {file.name}..."):
                    try:
                        content, err = read_file_content(file)
                        if err:
                            st.warning(f"Skipping {file.name}: {err}")
                            continue
                        summary_tuple = summarize_file_content(content, file.name)
                        if isinstance(summary_tuple, tuple):
                            summary = summary_tuple[0]
                            st.markdown(f"**Summary**: {summary}")
                            st.write("---")
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                    finally:
                        file.seek(0)

###############################################################################
# 6. HIERARCHICAL CLASSIFICATION WITH "CONTEXT AND KEYWORDS" APPROACH
###############################################################################

def hierarchical_query_page():
    """
    This page demonstrates a context-based approach:
    We have a dictionary of 'typical keywords' for each category. 
    We'll pass them to the LLM so it can cross-reference whether 
    the doc content matches that category well.
    """
    st.title("Hierarchical Query Path Generator")

    # Basic session state
    if 'hierarchy_depth' not in st.session_state:
        st.session_state.hierarchy_depth = None
    if 'hierarchy' not in st.session_state:
        st.session_state.hierarchy = {}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'typical_keywords' not in st.session_state:
        # A sample dictionary of typical words for each category:
        # You can expand these or alter them to match your domain
        st.session_state.typical_keywords = {
            "Cars": ["car", "vehicle", "engine", "automobile", "driving"],
            "Cars/European": ["mercedes", "bmw", "audi", "renault", "peugeot"],
            "Cars/Japanese": ["toyota", "honda", "nissan", "mazda"],
            "Planes": ["airplane", "airport", "aircraft", "flight", "pilot"],
            "Planes/Commercial": ["airbus", "boeing", "passenger", "airlines"],
            "Planes/Military": ["fighter", "f35", "stealth", "missile", "warplane"]
        }

    # Two columns
    left_col, right_col = st.columns([2,3])
    
    with left_col:
        st.markdown("""
        ### Steps:
        1. **Set Depth**
        2. **Define Hierarchy + typical keywords** 
        3. **Upload Documents**
        4. **Generate** (classify them)
        """)
        
        # Step 1: Depth
        st.markdown("#### Step 1: Set Depth")
        if st.session_state.hierarchy_depth is None:
            depth = st.select_slider("Select hierarchy depth", options=[2,3,4], value=2)
            if st.button("Initialize Hierarchy"):
                st.session_state.hierarchy_depth = depth
                st.session_state.hierarchy = {}
                st.rerun()
        else:
            st.info(f"Current depth: {st.session_state.hierarchy_depth} levels")
            if st.button("Reset Hierarchy"):
                st.session_state.hierarchy_depth = None
                st.session_state.hierarchy = {}
                st.rerun()

        # Step 3: Upload
        st.markdown("#### Step 3: Upload Documents")
        uf = st.file_uploader("Choose multiple docs", accept_multiple_files=True, type=['pdf','txt','docx'])
        if uf:
            st.session_state.uploaded_files = uf
            st.success(f"Uploaded {len(uf)} documents.")
        
        # Step 4: Generate
        st.markdown("#### Step 4: Generate Hierarchy")
        if st.session_state.hierarchy and st.session_state.uploaded_files:
            if st.button("Generate Hierarchy"):
                with st.spinner("Classifying..."):
                    try:
                        results = process_files_for_hierarchy(
                            st.session_state.uploaded_files,
                            st.session_state.hierarchy,
                            st.session_state.typical_keywords
                        )
                        st.session_state.hierarchy_results = results
                        st.success("Done!")
                        
                        # Visualization
                        nodes, edges, config = visualize_hierarchy_results(results)
                        st.markdown("### Hierarchy Visualization")
                        agraph(nodes=nodes, edges=edges, config=config)

                        # Export
                        st.markdown("### Export Results")
                        csv_data = export_hierarchy_results(results)
                        st.download_button(
                            "Download CSV",
                            data=csv_data,
                            file_name="hierarchy_results.csv",
                            mime="text/csv"
                        )

                        # Detailed
                        st.markdown("### Detailed Results")
                        for doc_name, doc_info in results.items():
                            with st.expander(f"📄 {doc_name}"):
                                st.write(f"**Category Path:** {doc_info['path']}")
                                st.write(f"**Preview:** {doc_info['content']}")
                    except Exception as e:
                        st.error(f"Error generating hierarchy: {str(e)}")
        else:
            st.warning("Complete steps 1-3 first.")
    
    with right_col:
        # Step 2: Category + typical words
        if st.session_state.hierarchy_depth is not None:
            st.markdown("#### Step 2: Define Hierarchy + Keywords")
            level_tabs = st.tabs([f"Level {i+1}" for i in range(st.session_state.hierarchy_depth)])
            
            def add_category(parent_path="", level=1):
                with level_tabs[level-1]:
                    if parent_path:
                        st.markdown(f"*Path so far: {parent_path}*")
                    col1, col2 = st.columns([3,1])
                    with col1:
                        new_cat = st.text_input("New subcategory", key=f"new_cat_{parent_path}_{level}")
                    with col2:
                        if st.button("Add", key=f"add_{parent_path}_{level}"):
                            cur_level = st.session_state.hierarchy
                            parts = parent_path.split("/") if parent_path else []
                            for p in parts:
                                if p:
                                    cur_level = cur_level[p]
                            if isinstance(cur_level, dict):
                                if new_cat not in cur_level:
                                    # If not last level, init as dict
                                    # If last, can still do dict
                                    if level < st.session_state.hierarchy_depth:
                                        cur_level[new_cat] = {}
                                    else:
                                        cur_level[new_cat] = {}
                                    st.success(f"Added {new_cat}")
                                    st.rerun()
                    
                    # Show existing
                    if st.session_state.hierarchy:
                        st.markdown("##### Current Subcategories:")
                        cur_level = st.session_state.hierarchy
                        for p in parent_path.split("/"):
                            if p:
                                cur_level = cur_level[p]
                        if isinstance(cur_level, dict):
                            for ckey in list(cur_level.keys()):
                                cc1, cc2 = st.columns([4,1])
                                with cc1:
                                    st.markdown(f"🔹 {ckey}")
                                with cc2:
                                    if st.button("🗑️", key=f"del_{parent_path}_{ckey}"):
                                        del cur_level[ckey]
                                        st.rerun()
                                # Recurse deeper
                                if level < st.session_state.hierarchy_depth:
                                    new_p = f"{parent_path}/{ckey}" if parent_path else ckey
                                    add_category(new_p, level+1)

            add_category()
            
            st.markdown("#### Complete Hierarchy Preview")
            if st.session_state.hierarchy:
                def display_hier(data, lvl=0):
                    out = ""
                    indent = "   " * lvl
                    for cat, subs in data.items():
                        out += f"{indent}📁 {cat}\n"
                        if isinstance(subs, dict):
                            out += display_hier(subs, lvl+1)
                    return out
                st.code(display_hier(st.session_state.hierarchy), language="plaintext")
            else:
                st.info("Add subcategories above to see preview.")

            # Let user also **edit** typical keywords dictionary in the sidebar
            st.markdown("### Typical Keywords Dictionary")
            st.write("""
                Each category can have known 'typical words' that appear in a relevant file. 
                E.g. 'BMW' or 'Mercedes' for 'Cars/European', 'Airbus' or 'Boeing' for 'Planes/Commercial'.
            """)
            st.write("These words are used in the LLM prompt to measure how well a file matches the category.")
            st.write("Current dictionary:")
            for k, words in st.session_state.typical_keywords.items():
                st.markdown(f"- **{k}**: {words}")

###############################################################################
# 7. SCORING FUNCTION THAT USES 'TYPICAL WORDS' AND PARENT CONTEXT
###############################################################################

def score_category_with_keywords(content, category_path, typical_keywords, threshold=0.3):
    """
    This function uses:
      1) The parent's path context
      2) A known list of 'typical words' for the current category path
    And asks the LLM: "Given these keywords, how well does the doc match this category?"

    We return a float score [0.0 ... 1.0].
    """
    # Extract typical words if present, else empty
    # e.g. if category_path = "Cars/European", we see if there's an entry in typical_keywords
    words_for_cat = typical_keywords.get(category_path, [])
    parent_context = category_path.rsplit("/", 1)[0] if "/" in category_path else ""
    
    # Build prompt
    if parent_context:
        context_str = f"This category is nested under: '{parent_context}'.\n"
    else:
        context_str = "This category is at the root level.\n"

    prompt = f"""
    You are classifying a document within a hierarchical taxonomy.

    Category path: {category_path}
    {context_str}

    Typical words for this category: {', '.join(words_for_cat)}.

    Rate how well this document fits the above category (0.0 to 1.0):
    - 1.0 = Perfect match (document strongly features the typical words & concepts)
    - 0.0 = No match at all

    Document excerpt (up to 1000 chars):
    {content[:1000]}

    Return ONLY the numeric score (no words, no explanation).
    """
    try:
        resp = llm_handler.get_response(prompt)
        resp = resp.strip().split()[0]  # just first token
        resp = ''.join(c for c in resp if c.isdigit() or c == '.')
        if resp:
            return float(resp)
        return 0.0
    except Exception as e:
        st.write(f"Error scoring category '{category_path}': {str(e)}")
        return 0.0

###############################################################################
# 8. RECURSIVE CLASSIFICATION USING TYPICAL WORDS
###############################################################################

def find_best_category_path(content, hierarchy, typical_keywords, current_path="", threshold=0.3):
    """
    Recursively find the best scoring category path.
    We build up 'current_path' as we go deeper.
    For each node:
      1) Get a numeric score from 'score_category_with_keywords'
      2) If it meets threshold, check subcategories
      3) If subcat ties or beats parent's score, prefer subcat
    """
    best_local_path = "Uncategorized"
    best_local_score = 0.0
    
    for category, subcats in hierarchy.items():
        # Build full path like "Cars", or "Cars/European", etc.
        new_path = f"{current_path}/{category}" if current_path else category
        
        # 1. Score this category path
        sc = score_category_with_keywords(content, new_path, typical_keywords, threshold)
        st.write(f"DEBUG: Path '{current_path or '(root)'}' → checking '{category}', score={sc:.4f}")

        candidate_path = new_path
        candidate_score = sc

        # 2. If above threshold and has subcats, go deeper
        if sc >= threshold and isinstance(subcats, dict) and len(subcats) > 0:
            child_path, child_score = find_best_category_path(content, subcats, typical_keywords, new_path, threshold)
            if child_score >= candidate_score:
                candidate_path = child_path
                candidate_score = child_score
        
        # 3. Compare with best so far
        if candidate_score > best_local_score:
            best_local_score = candidate_score
            best_local_path = candidate_path
    
    return best_local_path, best_local_score

def find_category_path(content, hierarchy, typical_keywords, threshold=0.3):
    """
    Wrapper to run the recursion. If best_score < 0.05, return 'Uncategorized'.
    """
    best_path, best_score = find_best_category_path(content, hierarchy, typical_keywords, "", threshold)
    st.write(f"DEBUG: Best path='{best_path}', best_score={best_score:.4f}, threshold={threshold}")
    if best_score < 0.05:
        return "Uncategorized"
    return best_path

def process_files_for_hierarchy(files, hierarchy, typical_keywords):
    """
    Classify each uploaded file using the multi-level approach 
    that cross-references typical keywords for each category path.
    """
    processed_results = {}
    for file in files:
        try:
            content, err = read_file_content(file)
            if err:
                continue
            st.write(f"---\n**Processing file:** {file.name}")
            
            path = find_category_path(content, hierarchy, typical_keywords, threshold=0.3)
            processed_results[file.name] = {
                'path': path,
                'content': content[:200] + "..."
            }
            st.write(f"DEBUG: Final chosen path for '{file.name}' = {path}")
            st.write("---")
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    return processed_results

###############################################################################
# 9. VISUALIZATION AND EXPORT
###############################################################################

def visualize_hierarchy_results(hierarchy_results):
    """Create a streamlit-agraph representation of the final classification."""
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
                target_id = f"cat_{len(parts)-1}_{last_cat}"
                edges.append(Edge(source=target_id, target=doc_id))
            else:
                # If uncategorized, link to special node
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
    
    # Add categories
    add_category_nodes(st.session_state.hierarchy)
    # Add docs
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
    """Export classification results to CSV."""
    import pandas as pd
    data = []
    for doc_name, doc_info in hierarchy_results.items():
        data.append({
            "Document": doc_name,
            "Category Path": doc_info["path"],
            "Preview": doc_info["content"]
        })
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

###############################################################################
# 10. RUN THE STREAMLIT APP
###############################################################################

if __name__ == "__main__":
    main()
