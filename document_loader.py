"""
Document Loader and Text Processor for Insurance RAG System
This file handles loading and preprocessing text files from the insurance documents folder.
"""

import os
import re
from typing import List, Dict, Tuple
from pathlib import Path

class DocumentLoader:
    def __init__(self, data_folder_path: str):
        """
        Initialize the document loader with the path to your text files folder.

        Args:
            data_folder_path: Path to the folder containing insurance text files
        """
        self.data_folder_path = Path(data_folder_path)
        self.documents = []

    def load_all_documents(self) -> List[Dict]:
        """
        Load all text files from the specified folder.

        Returns:
            List of dictionaries containing document content and metadata
        """
        documents = []

        # Check if folder exists
        if not self.data_folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.data_folder_path}")

        # Get all .txt files in the folder
        txt_files = list(self.data_folder_path.glob("*.txt"))

        if not txt_files:
            raise ValueError(f"No .txt files found in {self.data_folder_path}")

        print(f"Found {len(txt_files)} text files to process...")

        for file_path in txt_files:
            try:
                # Read the file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()

                # Skip empty files
                if not content.strip():
                    print(f"Skipping empty file: {file_path.name}")
                    continue

                # Create document dictionary
                document = {
                    'filename': file_path.name,
                    'file_path': str(file_path),
                    'content': content,
                    'processed_content': self.preprocess_text(content),
                    'word_count': len(content.split()),
                    'char_count': len(content)
                }

                documents.append(document)
                print(f"Loaded: {file_path.name} ({document['word_count']} words)")

                # Show cleaning progress for first few files
                if len(documents) <= 3:
                    print(f"  - Cleaned content preview: {document['processed_content'][:100]}...")

            except Exception as e:
                print(f"Error loading {file_path.name}: {str(e)}")
                continue

        self.documents = documents
        print(f"Successfully loaded {len(documents)} documents")
        return documents

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess the text content.

        Args:
            text: Raw text content

        Returns:
            Cleaned text content
        """
        # Remove specific unwanted words/phrases from insurance documents
        unwanted_words = [
            "FC & S EDITORS",
            "FC &S EDITORS",
            "FC& S EDITORS",
            "FC&S EDITORS",
            "CHANNELS",
            "RESOURCES",
            "Q&A",
            "CHARTS",
            "FORMS"
        ]

        # Remove unwanted words (case insensitive)
        for word in unwanted_words:
            text = re.sub(re.escape(word), '', text, flags=re.IGNORECASE)

        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Preserve dollar amount placeholders
        text = re.sub(r'\$\s*_+', '$ [AMOUNT]', text)

        # Clean up artifacts while preserving important characters
        text = re.sub(r'[^\w\s\$\[\]\(\).,;:!?%-]', '', text)

        # Remove lines that are just numbers or artifacts
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3 and not line.isdigit():
                # Check if line is just numbers, spaces, dashes, dots
                if not re.match(r'^[0-9\s\-\.]+$', line):
                    cleaned_lines.append(line)

        text = ' '.join(cleaned_lines)

        # Final cleanup
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def get_document_stats(self) -> Dict:
        """
        Get statistics about the loaded documents.

        Returns:
            Dictionary with document statistics
        """
        if not self.documents:
            return {"error": "No documents loaded"}

        total_words = sum(doc['word_count'] for doc in self.documents)
        total_chars = sum(doc['char_count'] for doc in self.documents)
        avg_words_per_doc = total_words / len(self.documents)

        return {
            'total_documents': len(self.documents),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_document': round(avg_words_per_doc, 2),
            'largest_document': max(self.documents, key=lambda x: x['word_count']),
            'smallest_document': min(self.documents, key=lambda x: x['word_count'])
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the loader with your folder path
    folder_path = r"C:\Users\Yesh\PycharmProjects\insurance-rag-system\data\text_files"

    loader = DocumentLoader(folder_path)

    try:
        # Load all documents
        documents = loader.load_all_documents()

        # Print statistics
        stats = loader.get_document_stats()
        print("\n=== Document Statistics ===")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Total words: {stats['total_words']}")
        print(f"Average words per document: {stats['average_words_per_document']}")
        print(f"Largest document: {stats['largest_document']['filename']} ({stats['largest_document']['word_count']} words)")
        print(f"Smallest document: {stats['smallest_document']['filename']} ({stats['smallest_document']['word_count']} words)")

        # Show a sample of the first document
        if documents:
            print(f"\n=== Sample from {documents[0]['filename']} ===")
            print(documents[0]['processed_content'][:300] + "...")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your folder path and ensure it contains .txt files")