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
    st.markdown("""
    ### Hierarchical Query Path Generator
    This tool organizes your documents into a hierarchical structure using recursive clustering.
    """)
    
    # Initialize session state for this page
    if 'root_categories' not in st.session_state:
        st.session_state.root_categories = []
    
    # Root category management
    st.subheader("1. Define Root Categories")
    new_category = st.text_input("Add a root category (e.g., Cars, Planes)")
    if st.button("Add Root Category"):
        if new_category and new_category not in st.session_state.root_categories:
            st.session_state.root_categories.append(new_category)
            st.success(f"Added root category: {new_category}")
    
    # Display and manage root categories
    if st.session_state.root_categories:
        st.write("Current root categories:")
        for idx, category in enumerate(st.session_state.root_categories):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"• {category}")
            with col2:
                if st.button("Remove", key=f"remove_{idx}"):
                    st.session_state.root_categories.pop(idx)
                    st.rerun()
    
    # Clustering parameters
    st.subheader("2. Configure Clustering")
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider("Maximum hierarchy depth", 2, 6, 4)
        min_cluster_size = st.slider("Minimum cluster size", 2, 10, 3)
    with col2:
        n_clusters_per_level = st.slider("Max clusters per level", 2, 8, 4)
    
    # File upload
    st.subheader("3. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose documents to cluster", 
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )
    
    # Process and cluster
    if uploaded_files and st.session_state.root_categories:
        if st.button("Generate Hierarchy"):
            with st.spinner("Processing documents and generating hierarchy..."):
                try:
                    # Process documents
                    documents = process_documents(uploaded_files)
                    
                    # Generate clusters
                    hierarchy = generate_hierarchy(
                        documents,
                        st.session_state.root_categories,
                        max_depth,
                        min_cluster_size,
                        n_clusters_per_level
                    )
                    
                    # Display results
                    st.subheader("Generated Hierarchy")
                    display_hierarchy(hierarchy)
                    
                except Exception as e:
                    st.error(f"Error generating hierarchy: {str(e)}")
    elif not st.session_state.root_categories:
        st.warning("Please add at least one root category before processing.")
    elif not uploaded_files:
        st.warning("Please upload documents to process.")

def process_documents(files):
    """Process uploaded documents and extract text content"""
    documents = []
    for file in files:
        content, error = read_file_content(file)
        if error:
            st.warning(f"Skipping {file.name}: {error}")
            continue
        documents.append({
            'name': file.name,
            'content': content
        })
        file.seek(0)
    return documents

def generate_hierarchy(documents, root_categories, max_depth, min_cluster_size, n_clusters_per_level):
    """Generate hierarchical clusters"""
    # Convert documents to TF-IDF vectors with better parameters for our use case
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    # Create document vectors
    doc_vectors = vectorizer.fit_transform([doc['content'] for doc in documents])
    
    # Create root category vectors with more context
    root_category_texts = []
    for category in root_categories:
        # Add more context to each category
        if category.lower() == 'cars':
            context = """cars vehicles automobile automotive motor vehicle sedan SUV 
                        Toyota Honda BMW Mercedes engine wheels driving"""
        elif category.lower() == 'planes':
            context = """planes aircraft aviation airplane jets flying flight 
                        Boeing Airbus aircraft aviation aerospace flying pilot"""
        root_category_texts.append(context)
    
    category_vectors = vectorizer.transform(root_category_texts)
    
    # Initial clustering by root categories
    hierarchy = {category: [] for category in root_categories}
    
    # Assign documents to root categories with improved similarity calculation
    for idx, doc in enumerate(documents):
        doc_vector = doc_vectors[idx]
        # Calculate similarities
        similarities = (doc_vector @ category_vectors.T).toarray().flatten()
        category_idx = np.argmax(similarities)
        category = root_categories[category_idx]
        hierarchy[category].append(doc)
    
    def cluster_level(docs, current_depth=0):
        if (current_depth >= max_depth or 
            len(docs) < min_cluster_size * 2):
            return {'documents': docs}
        
        # Create vectors for this subset of documents
        local_vectors = vectorizer.transform([d['content'] for d in docs])
        
        # Determine number of clusters
        n_clusters = min(len(docs) // min_cluster_size, n_clusters_per_level)
        if n_clusters < 2:
            return {'documents': docs}
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(local_vectors)
        
        # Organize documents by cluster
        clusters = defaultdict(list)
        for i, doc in enumerate(docs):
            clusters[labels[i]].append(doc)
        
        # Recursively cluster each group
        result = {}
        for label, cluster_docs in clusters.items():
            try:
                # Generate cluster name
                if current_depth > 0:
                    terms = get_cluster_terms(cluster_docs, vectorizer)
                    cluster_name = f"Topics: {terms}"
                else:
                    cluster_name = f"Group {label + 1}"
                
                result[cluster_name] = cluster_level(
                    cluster_docs,
                    current_depth + 1
                )
            except Exception as e:
                st.error(f"Error processing cluster: {str(e)}")
                cluster_name = f"Group {label + 1}"
                result[cluster_name] = {'documents': cluster_docs}
        
        return result
    
    # Process each root category
    for category in root_categories:
        if len(hierarchy[category]) > min_cluster_size:
            hierarchy[category] = cluster_level(hierarchy[category], 1)
        else:
            hierarchy[category] = {'documents': hierarchy[category]}
    
    # Print debug information
    st.write("Document distribution across root categories:")
    for category in root_categories:
        doc_count = len(hierarchy[category].get('documents', []))
        if isinstance(hierarchy[category], dict):
            for subcat in hierarchy[category].values():
                if isinstance(subcat, dict):
                    doc_count += len(subcat.get('documents', []))
        st.write(f"{category}: {doc_count} documents")
    
    return hierarchy

def display_hierarchy(hierarchy):
    """Display the hierarchical structure as an interactive graph"""
    nodes = []
    edges = []
    
    def process_level(data, parent_id=None, level=0):
        for category, content in data.items():
            # Create unique ID for this node
            current_id = f"{level}_{category}"
            
            # Add node
            nodes.append(Node(
                id=current_id,
                label=category,
                size=20,
                shape="dot" if isinstance(content, dict) else "square"
            ))
            
            # Add edge from parent if exists
            if parent_id:
                edges.append(Edge(source=parent_id, target=current_id))
            
            if isinstance(content, dict):
                # Recursive call for nested categories
                if 'documents' in content:
                    # Add document nodes
                    for doc in content['documents']:
                        doc_id = f"doc_{doc['name']}"
                        nodes.append(Node(
                            id=doc_id,
                            label=doc['name'],
                            size=15,
                            shape="square",
                            color="#1f77b4"
                        ))
                        edges.append(Edge(source=current_id, target=doc_id))
                else:
                    # Process next level
                    process_level(content, current_id, level + 1)
            else:
                # Add document nodes for leaf categories
                for doc in content:
                    doc_id = f"doc_{doc['name']}"
                    nodes.append(Node(
                        id=doc_id,
                        label=doc['name'],
                        size=15,
                        shape="square",
                        color="#1f77b4"
                    ))
                    edges.append(Edge(source=current_id, target=doc_id))
    
    # Process the hierarchy
    process_level(hierarchy)
    
    # Configure the graph
    config = Config(
        width=750,
        height=950,
        directed=True,
        physics=True,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': False}
    )
    
    # Display the graph
    st.write("Click and drag nodes to explore the hierarchy:")
    agraph(nodes=nodes, edges=edges, config=config)

def get_cluster_terms(docs, vectorizer, n_terms=3):
    """Get most representative terms for a cluster"""
    try:
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get TF-IDF matrix for cluster documents
        tfidf_matrix = vectorizer.transform([d['content'] for d in docs])
        
        # Calculate average TF-IDF scores across documents
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get indices of top terms
        top_indices = np.argsort(avg_tfidf)[-n_terms:][::-1]
        
        # Get top terms
        top_terms = [feature_names[i] for i in top_indices]
        
        return ", ".join(top_terms)
    except Exception as e:
        st.error(f"Error generating cluster terms: {str(e)}")
        return f"Group {hash(str(docs))[:5]}"  # Fallback cluster name

if __name__ == "__main__":
    main()
