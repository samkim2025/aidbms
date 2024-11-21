import chromadb

class DatabaseHandler:
    def __init__(self):
        # Update the client initialization to use the new syntax
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Keep track of category structure
        self.category_tree = {}

    def add_document(self, document):
        """Add a document to the database"""
        try:
            self.collection.add(
                documents=[document['content']],
                metadatas=[{
                    'title': document['title'],
                    'file_type': document['file_type'],
                    'categories': json.dumps([])  # Will be updated later
                }],
                ids=[document['id']]
            )
            return True
        except Exception as e:
            print(f"Error adding document to database: {str(e)}")
            return False

    def update_document_categories(self, doc_id, categories):
        """Update categories for a document"""
        self.collection.update(
            ids=[doc_id],
            metadatas=[{'categories': json.dumps(categories)}]
        )

    def get_filtered_documents(self, category_path):
        """Retrieve documents matching the category path"""
        if not category_path:
            return self.get_all_documents()
        
        # Convert category path to string for matching
        category_string = json.dumps(category_path)
        
        # Query documents with matching category path
        results = self.collection.query(
            query_texts=[""],
            where={"categories": {"$contains": category_string}},
            n_results=100
        )
        
        return self._format_results(results)

    def get_all_documents(self):
        """Retrieve all documents"""
        results = self.collection.get()
        return self._format_results(results)

    def _format_results(self, results):
        """Format database results into a clean structure"""
        formatted_docs = []
        
        if not results or not results['ids']:
            return formatted_docs
        
        for i in range(len(results['ids'])):
            formatted_docs.append({
                'id': results['ids'][i],
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return formatted_docs

    def update_category_tree(self, tree):
        """Update the category tree structure"""
        self.category_tree = tree

    def get_category_tree(self):
        """Get the current category tree"""
        return self.category_tree