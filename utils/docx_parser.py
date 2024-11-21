import io
import zipfile
import xml.etree.ElementTree as ET
from typing import Optional

class DocxParser:
    @staticmethod
    def parse(file_obj) -> str:
        """Parse content from a .docx file object"""
        try:
            # Create a bytes buffer from the file object
            buffer = io.BytesIO(file_obj.read())
            
            text = []
            # Open the .docx as a zip file
            with zipfile.ZipFile(buffer) as docx:
                # Read the document.xml file from the zip
                xml_content = docx.read('word/document.xml')
                
                # Parse the XML
                tree = ET.fromstring(xml_content)
                
                # Find all paragraphs (w:p elements)
                # Define the namespace
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                for paragraph in tree.findall('.//w:p', ns):
                    # Get all text elements (w:t) in the paragraph
                    text_elements = paragraph.findall('.//w:t', ns)
                    if text_elements:
                        # Combine all text elements in the paragraph
                        paragraph_text = ''.join(element.text or '' for element in text_elements)
                        text.append(paragraph_text)
            
            return '\n'.join(text)
            
        except Exception as e:
            print(f"Error parsing DOCX file: {str(e)}")
            return "" 