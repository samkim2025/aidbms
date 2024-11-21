import re
from pathlib import Path
from typing import Optional

class TextParser:
    @staticmethod
    def parse(file_obj) -> str:
        """Parse text content from a file object"""
        try:
            content = file_obj.read().decode('utf-8')
            return TextParser._process_content(content)
        except Exception as e:
            print(f"Error parsing text file: {str(e)}")
            return ""
    
    @staticmethod
    def _process_content(content: str) -> str:
        """Process the content through all cleaning steps"""
        content = TextParser._normalize_encoding(content)
        content = TextParser._normalize_line_endings(content)
        content = TextParser._remove_unnecessary_characters(content)
        content = TextParser._normalize_spaces(content)
        content = TextParser._remove_common_patterns(content)
        content = TextParser._additional_cleaning_steps(content)
        return content
    
    @staticmethod
    def _normalize_encoding(content: str) -> str:
        """Normalize text encoding to UTF-8"""
        try:
            # In Python, strings are already Unicode, so we just ensure it's valid UTF-8
            return content.encode('utf-8').decode('utf-8')
        except Exception:
            return content
    
    @staticmethod
    def _normalize_line_endings(content: str) -> str:
        """Normalize all line endings to \n"""
        return content.replace('\r\n', '\n').replace('\r', '\n')
    
    @staticmethod
    def _remove_unnecessary_characters(content: str) -> str:
        """Remove all characters except alphanumeric and basic punctuation"""
        return re.sub(r'[^a-zA-Z0-9\s\.,;:?!()\-_]', '', content)
    
    @staticmethod
    def _normalize_spaces(content: str) -> str:
        """Normalize multiple spaces to single space"""
        return re.sub(r'\s+', ' ', content).strip()
    
    @staticmethod
    def _remove_common_patterns(content: str) -> str:
        """Remove URLs, email addresses, and comments"""
        # Remove URLs
        content = re.sub(r'http[^\s]+', '', content)
        # Remove email addresses
        content = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', content)
        # Remove C-style comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return content
    
    @staticmethod
    def _additional_cleaning_steps(content: str) -> str:
        """Perform additional cleaning steps"""
        # Remove digits
        content = re.sub(r'\d', '', content)
        # Normalize multiple newlines
        content = re.sub(r'\n+', '\n', content)
        return content 