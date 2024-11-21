from docx import Document
import io

class DocxParser:
    @staticmethod
    def parse(file_obj):
        """Parse content from a .docx file object"""
        try:
            # Create a bytes buffer from the file object
            file_bytes = file_obj.read()
            file_obj.seek(0)  # Reset file pointer
            
            # Create a document object from the bytes
            doc = Document(io.BytesIO(file_bytes))
            
            # Extract text from paragraphs
            text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text.strip())
            
            return "\n".join(text)
            
        except Exception as e:
            print(f"Error parsing DOCX: {str(e)}")
            return "" 