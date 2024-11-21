import json

class AICategorizer:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
        self.category_tree = {}
        self.categories = set()
        self.load_categories()

    async def generate_categories(self, documents):
        """Generate hierarchical categories based on document content"""
        # Prepare document summaries for LLM
        doc_summaries = [
            f"Document: {doc['title']}\nContent: {doc['content'][:500]}..."
            for doc in documents[:10]  # Limit to first 10 docs for initial categorization
        ]
        
        # Get category suggestions from LLM
        system_prompt = """
        You are an expert document categorizer. Create an intuitive hierarchical 
        category structure that would help users navigate through these documents.
        The structure should be 3-4 levels deep and focus on practical navigation.
        """
        
        prompt = f"""
        Based on these documents:
        {doc_summaries}
        
        Create a hierarchical category structure. Return the result as a JSON object where:
        - Each key is a category name
        - Each value is either another object (for subcategories) or null (for leaf nodes)
        
        Example format:
        {{
            "Category1": {{
                "Subcategory1": {{
                    "SubSubcategory1": null
                }}
            }}
        }}
        """
        
        response = await self.llm_handler.get_response(prompt, system_prompt)
        
        try:
            self.category_tree = json.loads(response)
            return self.category_tree
        except json.JSONDecodeError:
            # Fallback to simple category structure if LLM response isn't valid JSON
            return self._generate_fallback_categories()

    def categorize_document(self, document):
        """Categorize a single document into the existing category tree"""
        prompt = f"""
        Given this document:
        Title: {document['title']}
        Content: {document['content'][:1000]}...
        
        And this category structure:
        {json.dumps(self.category_tree, indent=2)}
        
        Return a JSON array representing the most appropriate category path for this document.
        Example: ["Category1", "Subcategory1", "SubSubcategory1"]
        """
        
        response = self.llm_handler.get_completion(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    def get_categories_at_level(self, level, selected_path):
        """Get available categories at the specified level"""
        current_node = self.category_tree
        
        # Navigate to current position in tree
        for path_item in selected_path:
            if path_item in current_node:
                current_node = current_node[path_item]
            else:
                return []
        
        # Return available categories at current level
        if isinstance(current_node, dict):
            return list(current_node.keys())
        return []

    def _generate_fallback_categories(self):
        """Generate simple fallback categories if LLM categorization fails"""
        return {
            "Documents": {
                "Text Files": None,
                "PDFs": None,
                "Word Documents": None
            }
        }

    def load_categories(self):
        try:
            with open('categories.txt', 'r') as f:
                self.categories = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            self.categories = set()

    def save_categories(self):
        with open('categories.txt', 'w') as f:
            for category in sorted(self.categories):
                f.write(f"{category}\n")

    def get_categories(self):
        """Return list of current categories"""
        return sorted(list(self.categories))

    def add_category(self, category):
        """Add a new category"""
        self.categories.add(category)
        self.save_categories()

    def remove_category(self, category):
        """Remove a category"""
        if category in self.categories:
            self.categories.remove(category)
            self.save_categories()