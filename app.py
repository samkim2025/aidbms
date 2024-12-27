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
# SECTION 1: INITIALIZATION
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
    """Decorator for timeouts."""
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
    """Read file content with a 5-second timeout."""
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

###############################################################################
# SECTION 2: TOP-LEVEL CATEGORIZATION (DOCUMENT MANAGEMENT)
###############################################################################

@timeout_handler(5)
def categorize_file(content, categories):
    """
    Classify file content into exactly ONE of the top-level categories.
    Return the single best match or 'Uncategorized'.
    """
    if not categories or not content.strip():
        return "Uncategorized"
    
    prompt = f"""
    You are an expert classifier. Categorize this text into exactly ONE of these 
    top-level categories: {', '.join(categories)}.
    
    If it doesn't match well, return 'Uncategorized'.

    Document excerpt:
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
        print(f"categorize_file error: {str(e)}")
        return "Uncategorized"

def reclassify_all_documents():
    """Re-run top-level classification after categories change."""
    if not st.session_state.get('uploaded_files_names'):
        return
    with st.spinner("Reclassifying docs..."):
        old = st.session_state.file_categories.copy()
        for fname in st.session_state.uploaded_files_names:
            try:
                content = read_file_content(fname)
                if content and not isinstance(content, tuple):
                    new_cat, err = categorize_file(content, st.session_state.categories)
                    if not err:
                        st.session_state.file_categories[fname] = new_cat
                        if fname in old and old[fname] != new_cat:
                            st.info(f"Reclassified {fname}: {old[fname]} → {new_cat}")
            except Exception as e:
                st.error(f"Error reclassifying {fname}: {str(e)}")
                if fname in old:
                    st.session_state.file_categories[fname] = old[fname]
        st.success("Reclassification complete!")

###############################################################################
# SECTION 3: SUMMARIZE FILES
###############################################################################

@timeout_handler(10)
def summarize_file_content(content, file_name):
    """Generate a short summary for the file."""
    prompt = f"""
    Create a very brief summary (1-2 short sentences, under 200 chars) 
    for the document. Start with:
    "{file_name} is a {file_name.split('.')[-1]} file about..."

    Document excerpt:
    ---
    {content[:3000]}
    ---
    """
    return llm_handler.get_response(prompt)

###############################################################################
# SECTION 4: STREAMLIT APP
###############################################################################

def main():
    st.title("AI Document Management System")

    if 'categories' not in st.session_state:
        st.session_state.categories = []
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = set()
    if 'file_categories' not in st.session_state:
        st.session_state.file_categories = {}

    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose functionality",
            ["Document Management", "Hierarchical Query Generator"],
            index=0
        )
        # If on Document Management, show categories
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
                    st.warning("Please enter a category name!")
            
            if st.session_state.categories:
                st.subheader("Current Categories")
                for idx, cat in enumerate(st.session_state.categories):
                    c1, c2 = st.columns([3,1])
                    with c1:
                        editval = st.text_input(
                            f"Category {idx+1}",
                            value=cat,
                            key=f"edit_{idx}"
                        )
                        if editval != cat:
                            st.session_state.categories[idx] = editval
                            reclassify_all_documents()
                    with c2:
                        if st.button("🗑️", key=f"delete_{idx}"):
                            st.session_state.categories.pop(idx)
                            reclassify_all_documents()
                            st.rerun()
            else:
                st.info("No top-level categories yet. Add one above!")

    if page == "Document Management":
        document_management_page()
    else:
        hierarchical_query_page()


###############################################################################
# SECTION 5: DOCUMENT MANAGEMENT PAGE
###############################################################################

def document_management_page():
    """User can upload, parse, classify (top-level), summarize."""
    st.markdown("""
    ### Steps:
    1. **Upload** files.
    2. **Test Parse** (optional).
    3. **Classify** docs with existing categories.
    4. **Review** or **Remove**.
    5. **Summarize** if needed.
    """)
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload Documents", 
        accept_multiple_files=True,
        type=['pdf','txt','docx']
    )

    # Test parse
    if uploaded_files and st.button("Test Parse Files"):
        for f in uploaded_files:
            st.write(f"Testing parse for: {f.name}")
            content, err = read_file_content(f)
            if err:
                st.warning(f"Skipping {f.name}: {err}")
                continue
            st.write("First 200 chars:", content[:200])
            st.write("---")
            f.seek(0)

    # Classify
    if uploaded_files and st.button("Classify Documents"):
        with st.spinner("Classifying..."):
            for f in uploaded_files:
                if f.name not in st.session_state.uploaded_files_names:
                    content, err = read_file_content(f)
                    if err:
                        st.warning(f"Skipping {f.name}: {err}")
                        continue
                    cat, cat_err = categorize_file(content, st.session_state.categories)
                    if cat_err:
                        st.warning(f"Error: {cat_err}")
                        continue
                    st.session_state.file_categories[f.name] = cat
                    st.session_state.uploaded_files_names.add(f.name)
                    st.write(f"{f.name} → {cat}")
        st.success("Done!")

    # Show current docs
    if uploaded_files:
        st.subheader("Current Documents")
        for f in uploaded_files:
            c1, c2, c3, c4 = st.columns([2,1,2,1])
            with c1:
                st.write(f"📄 {f.name}")
            with c2:
                st.write(f"{f.size/1024:.1f} KB")
            with c3:
                curr = st.session_state.file_categories.get(f.name, "Uncategorized")
                catops = ["Uncategorized"] + st.session_state.categories
                try:
                    idx = catops.index(curr)
                except ValueError:
                    idx = 0
                newval = st.selectbox(
                    f"Category for {f.name}",
                    catops,
                    index=idx,
                    key=f"cat_{f.name}"
                )
            with c4:
                if st.button("Save", key=f"save_{f.name}"):
                    st.session_state.file_categories[f.name] = newval
                    st.session_state.uploaded_files_names.add(f.name)
                    st.success(f"Saved {f.name} → {newval}")
    
    # Show categorized
    st.header("Categorized Documents")
    if st.session_state.file_categories:
        all_cats = ["Uncategorized"] + st.session_state.categories
        tabs = st.tabs(all_cats)
        for tab, c in zip(tabs, all_cats):
            with tab:
                docsin = [fn for fn, cc in st.session_state.file_categories.items() if cc == c]
                if docsin:
                    for d in docsin:
                        colA, colB = st.columns([4,1])
                        with colA:
                            st.write(f"📄 {d}")
                        with colB:
                            if st.button("Remove", key=f"remove_{c}_{d}"):
                                del st.session_state.file_categories[d]
                                st.session_state.uploaded_files_names.remove(d)
                                st.rerun()
                else:
                    st.write("No documents in this category.")
    else:
        st.write("No docs have been classified yet.")

    # Clear
    if st.button("Clear All Documents"):
        st.session_state.file_categories.clear()
        st.session_state.uploaded_files_names.clear()
        st.rerun()

    # Summaries
    st.header("Summarize Files")
    if uploaded_files:
        for f in uploaded_files:
            st.write(f"📄 {f.name}")
            if st.button("Summarize", key=f"summarize_{f.name}"):
                with st.spinner(f"Summarizing {f.name}..."):
                    try:
                        content, err = read_file_content(f)
                        if err:
                            st.warning(f"Skipping {f.name}: {err}")
                            continue
                        summ_tuple = summarize_file_content(content, f.name)
                        if isinstance(summ_tuple, tuple):
                            summ = summ_tuple[0]
                            st.markdown(f"**Summary**: {summ}")
                            st.write("---")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    finally:
                        f.seek(0)

###############################################################################
# SECTION 6: TITLE-BASED HIERARCHICAL CLASSIFICATION
###############################################################################

def classify_by_filename(filename, hierarchy):
    """
    Title-based classification for the "Hierarchical Query Generator":
    We traverse the hierarchy and see which categories appear in the filename 
    (case-insensitive). We pick the **deepest** matching path we can find.
    If no match at all, return "Uncategorized".

    Example: if filename = "Commercial_Boeing_787.pdf" 
      - see if "Planes" is in name
      - see if "Commercial" is in name
      - see if "Boeing" is in name
    If it matches "Planes/Commercial/Boeing", that's the final path.
    """
    file_lower = filename.lower()
    
    best_path = "Uncategorized"
    best_depth = 0  # how many levels deep we matched
    stack = [(cat, 1, cat) for cat in hierarchy]  # (category_label, depth, path_str)
    
    while stack:
        label, depth, path_str = stack.pop()
        
        # If label appears in file name, we consider it matched
        if label.lower() in file_lower and depth > best_depth:
            best_depth = depth
            best_path = path_str
        
        # If subcategories exist, push them
        subcats = hierarchy.get(label)
        if isinstance(subcats, dict):
            for subcat in subcats:
                new_path = f"{path_str}/{subcat}"
                stack.append((subcat, depth + 1, new_path))
    
    return best_path


def process_files_for_hierarchy(files, hierarchy):
    """Use 'classify_by_filename' to assign each file to the deepest matching path."""
    results = {}
    for f in files:
        # We ignore file content for this approach
        best_path = classify_by_filename(f.name, hierarchy)
        excerpt = f.name  # We'll just store the name as "content preview"
        results[f.name] = {
            'path': best_path,
            'content': excerpt
        }
    return results

def visualize_hierarchy_results(hierarchy_results):
    """Create an agraph of the final classification."""
    nodes = []
    edges = []
    
    def add_category_nodes(hier, parent_id=None, level=0):
        for cat, subs in hier.items():
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
    
    def add_docs(results):
        for docn, info in results.items():
            doc_id = f"doc_{docn}"
            nodes.append(Node(
                id=doc_id,
                label=docn,
                size=15,
                color="#99ff99",
                shape="square"
            ))
            if info['path'] != "Uncategorized":
                parts = info['path'].split('/')
                final_cat = parts[-1]
                target_id = f"cat_{len(parts)-1}_{final_cat}"
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
    
    add_category_nodes(st.session_state.hierarchy)
    add_docs(hierarchy_results)
    
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
    for doc_name, info in hierarchy_results.items():
        data.append({
            "Document": doc_name,
            "Category Path": info["path"],
            "Preview": info["content"]
        })
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

###############################################################################
# SECTION 7: HIERARCHICAL QUERY PAGE
###############################################################################

def hierarchical_query_page():
    """
    A 'title-based' approach to classification for the Hierarchical Query Generator:
    We ignore file content and rely on the filename containing keywords 
    that match categories or subcategories in the hierarchy.
    """
    st.title("Hierarchical Query Path Generator")

    if 'hierarchy_depth' not in st.session_state:
        st.session_state.hierarchy_depth = None
    if 'hierarchy' not in st.session_state:
        st.session_state.hierarchy = {}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    left, right = st.columns([2,3])

    with left:
        st.markdown("""
        ### Steps:
        1. Set the depth
        2. Define your category hierarchy
        3. Upload documents
        4. Generate
        """)
        
        # Step 1: Set Depth
        st.markdown("#### Step 1: Set Hierarchy Depth")
        if st.session_state.hierarchy_depth is None:
            d = st.select_slider("Select depth", [2,3,4], value=2)
            if st.button("Initialize Hierarchy"):
                st.session_state.hierarchy_depth = d
                st.session_state.hierarchy = {}
                st.rerun()
        else:
            st.info(f"Current depth: {st.session_state.hierarchy_depth}")
            if st.button("Reset Hierarchy"):
                st.session_state.hierarchy_depth = None
                st.session_state.hierarchy = {}
                st.rerun()

        # Step 3: File Upload
        st.markdown("#### Step 3: Upload Documents")
        uf = st.file_uploader("Choose docs for classification", accept_multiple_files=True, type=['pdf','txt','docx'])
        if uf:
            st.session_state.uploaded_files = uf
            st.success(f"Uploaded {len(uf)} docs.")
        
        # Step 4: Generate
        st.markdown("#### Step 4: Generate Hierarchy")
        if st.session_state.hierarchy and st.session_state.uploaded_files:
            if st.button("Generate Hierarchy"):
                with st.spinner("Classifying by filename..."):
                    try:
                        results = process_files_for_hierarchy(
                            st.session_state.uploaded_files,
                            st.session_state.hierarchy
                        )
                        st.session_state.hierarchy_results = results
                        st.success("Done!")
                        
                        # Visualization
                        nodes, edges, config = visualize_hierarchy_results(results)
                        st.markdown("### Hierarchy Visualization")
                        agraph(nodes=nodes, edges=edges, config=config)

                        # Export
                        st.markdown("### Export Results")
                        csvdat = export_hierarchy_results(results)
                        st.download_button(
                            "Download CSV",
                            data=csvdat,
                            file_name="hierarchy_results.csv",
                            mime="text/csv"
                        )
                        
                        # Detailed
                        st.markdown("### Detailed Results")
                        for docn, docinfo in results.items():
                            with st.expander(f"📄 {docn}"):
                                st.write(f"**Category Path:** {docinfo['path']}")
                                st.write(f"**Preview:** {docinfo['content']}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("Complete steps 1-3 first.")

    with right:
        if st.session_state.hierarchy_depth is not None:
            st.markdown("#### Step 2: Define Categories")
            level_tabs = st.tabs([f"Level {i+1}" for i in range(st.session_state.hierarchy_depth)])
            
            def add_category(parent_path="", level=1):
                with level_tabs[level-1]:
                    if parent_path:
                        st.markdown(f"*Current path: {parent_path}*")
                    cA, cB = st.columns([3,1])
                    with cA:
                        newcat = st.text_input("Category name", key=f"new_cat_{parent_path}_{level}")
                    with cB:
                        if st.button("Add", key=f"add_{parent_path}_{level}"):
                            cur_level = st.session_state.hierarchy
                            parts = parent_path.split("/") if parent_path else []
                            for p in parts:
                                if p:
                                    cur_level = cur_level[p]
                            if isinstance(cur_level, dict):
                                if newcat not in cur_level:
                                    if level < st.session_state.hierarchy_depth:
                                        cur_level[newcat] = {}
                                    else:
                                        cur_level[newcat] = {}
                                    st.success(f"Added {newcat}")
                                    st.rerun()
                    
                    if st.session_state.hierarchy:
                        st.markdown("##### Current Categories:")
                        cur_level = st.session_state.hierarchy
                        for p in parent_path.split("/"):
                            if p:
                                cur_level = cur_level[p]
                        if isinstance(cur_level, dict):
                            for ckey in list(cur_level.keys()):
                                c1, c2 = st.columns([4,1])
                                with c1:
                                    st.markdown(f"🔹 {ckey}")
                                with c2:
                                    if st.button("🗑️", key=f"del_{parent_path}_{ckey}"):
                                        del cur_level[ckey]
                                        st.rerun()
                                # Recursively handle deeper if needed
                                if level < st.session_state.hierarchy_depth:
                                    newp = f"{parent_path}/{ckey}" if parent_path else ckey
                                    add_category(newp, level+1)

            add_category()

            # Show the final hierarchy preview
            st.markdown("#### Complete Hierarchy Preview")
            if st.session_state.hierarchy:
                def display_hier(hier, lvl=0):
                    out = ""
                    indent = "  " * lvl
                    for cat, subs in hier.items():
                        out += f"{indent}📁 {cat}\n"
                        if isinstance(subs, dict):
                            out += display_hier(subs, lvl+1)
                    return out
                st.code(display_hier(st.session_state.hierarchy), language="plaintext")
            else:
                st.info("Add subcategories to see the preview.")

###############################################################################
# SECTION 8: RUN STREAMLIT
###############################################################################

if __name__ == "__main__":
    main()
