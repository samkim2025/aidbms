import PyPDF2
import io

class PdfParser:
    @staticmethod
    def parse(file_obj):
        """Parse content from a PDF file object"""
        try:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file_obj)
            
            # Extract text from all pages
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            
            return "\n".join(text)
            
        except Exception as e:
            print(f"Error parsing PDF: {str(e)}")
            return "" 