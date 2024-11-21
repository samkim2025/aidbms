import PyPDF2
from docx import Document
import hashlib
import os
import tempfile

class DocumentParser:
    def __init__(self):
        self.supported_extensions = {
            'pdf': self._parse_pdf,
            'txt': self._parse_txt,
            'docx': self._parse_docx,
            'doc': self._parse_docx
        }

    def parse(self, file):
        """Parse uploaded file and extract content"""
        # Get file extension
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Parse the file using appropriate parser
            content = self.supported_extensions[file_extension](tmp_file_path)
            
            # Generate document ID
            doc_id = self._generate_document_id(content)
            
            return {
                'id': doc_id,
                'title': file.name,
                'content': content,
                'file_type': file_extension
            }
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    def _parse_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    def _parse_txt(self, file_path):
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def _parse_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Error parsing DOCX file: {str(e)}")

    def _generate_document_id(self, content):
        """Generate unique document ID based on content"""
        return hashlib.md5(content.encode()).hexdigest()