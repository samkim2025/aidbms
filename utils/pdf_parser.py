import PyPDF2
import io
import fitz  # PyMuPDF
import pdfplumber
import logging

class PdfParser:
    @staticmethod
    def parse(file_obj):
        """Parse PDF using multiple methods and return the best result"""
        file_content = file_obj.read()
        file_obj.seek(0)  # Reset file pointer
        
        methods = [
            ("PyPDF2", PdfParser._parse_with_pypdf2),
            ("PyMuPDF", PdfParser._parse_with_pymupdf),
            ("pdfplumber", PdfParser._parse_with_pdfplumber)
        ]
        
        # Try each parsing method
        for method_name, parser in methods:
            try:
                content = parser(file_content)
                if content and len(content.strip()) > 100:  # Verify we got meaningful content
                    print(f"Successfully parsed with {method_name}")
                    return content
                else:
                    print(f"{method_name} extracted no meaningful content")
            except Exception as e:
                print(f"Error with {method_name}: {str(e)}")
        
        return ""  # Return empty string if all methods fail

    @staticmethod
    def _parse_with_pypdf2(file_content):
        """Parse PDF using PyPDF2"""
        try:
            pdf = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = []
            for page in pdf.pages:
                try:
                    text.append(page.extract_text())
                except Exception as e:
                    print(f"PyPDF2 page extraction error: {str(e)}")
            return "\n".join(text)
        except Exception as e:
            print(f"PyPDF2 parsing error: {str(e)}")
            return ""

    @staticmethod
    def _parse_with_pymupdf(file_content):
        """Parse PDF using PyMuPDF (fitz)"""
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = []
            for page in doc:
                try:
                    text.append(page.get_text())
                except Exception as e:
                    print(f"PyMuPDF page extraction error: {str(e)}")
            doc.close()
            return "\n".join(text)
        except Exception as e:
            print(f"PyMuPDF parsing error: {str(e)}")
            return ""

    @staticmethod
    def _parse_with_pdfplumber(file_content):
        """Parse PDF using pdfplumber"""
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                text = []
                for page in pdf.pages:
                    try:
                        text.append(page.extract_text())
                    except Exception as e:
                        print(f"pdfplumber page extraction error: {str(e)}")
                return "\n".join(text)
        except Exception as e:
            print(f"pdfplumber parsing error: {str(e)}")
            return "" 