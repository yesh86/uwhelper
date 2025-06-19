"""
Text Chunker for Insurance RAG System
This file handles chunking documents into smaller pieces for better RAG performance.
"""

import re
from typing import List, Dict, Tuple
from document_loader import DocumentLoader


class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk all documents into smaller pieces.

        Args:
            documents: List of document dictionaries from DocumentLoader

        Returns:
            List of chunk dictionaries
        """
        all_chunks = []

        print(f"Chunking {len(documents)} documents...")
        print(f"Chunk size: {self.chunk_size} characters, Overlap: {self.overlap} characters")

        for doc_idx, document in enumerate(documents):
            # Get the processed content
            content = document['processed_content']

            if len(content) <= self.chunk_size:
                # Document is small enough to be one chunk
                chunk = {
                    'chunk_id': f"{document['filename']}_chunk_0",
                    'document_name': document['filename'],
                    'chunk_index': 0,
                    'content': content,
                    'char_count': len(content),
                    'word_count': len(content.split()),
                    'source_file': document['file_path']
                }
                all_chunks.append(chunk)
            else:
                # Split document into multiple chunks
                doc_chunks = self._split_document(document)
                all_chunks.extend(doc_chunks)

            # Progress update
            if (doc_idx + 1) % 100 == 0:
                print(f"Processed {doc_idx + 1}/{len(documents)} documents...")

        self.chunks = all_chunks
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def _split_document(self, document: Dict) -> List[Dict]:
        """
        Split a single document into chunks with smart sentence boundaries.

        Args:
            document: Single document dictionary

        Returns:
            List of chunk dictionaries for this document
        """
        content = document['processed_content']
        chunks = []

        # Try to split at sentence boundaries first
        sentences = self._split_into_sentences(content)

        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = self._create_chunk(
                    document, current_chunk, chunk_index
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk += sentence

        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                document, current_chunk, chunk_index
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling insurance-specific patterns.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Insurance documents often have numbered sections, so we need to be careful
        # Split on periods, but avoid splitting on common abbreviations

        # Common insurance abbreviations that shouldn't trigger sentence breaks
        abbreviations = [
            r'Co\.',  # Company
            r'Inc\.',  # Incorporated
            r'Ltd\.',  # Limited
            r'Corp\.',  # Corporation
            r'vs\.',  # versus
            r'etc\.',  # etcetera
            r'e\.g\.',  # for example
            r'i\.e\.',  # that is
            r'No\.',  # Number
            r'Art\.',  # Article
            r'Sec\.',  # Section
            r'Para\.',  # Paragraph
        ]

        # Temporarily replace abbreviations
        temp_text = text
        replacements = {}
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBREV_{i}__"
            temp_text = re.sub(abbrev, placeholder, temp_text, flags=re.IGNORECASE)
            replacements[placeholder] = abbrev.replace('\\', '')

        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', temp_text)

        # Restore abbreviations and clean up
        final_sentences = []
        for sentence in sentences:
            # Restore abbreviations
            for placeholder, original in replacements.items():
                sentence = sentence.replace(placeholder, original)

            # Clean and add sentence
            sentence = sentence.strip()
            if sentence:
                # Add back the sentence ending if it's not there
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '. '
                else:
                    sentence += ' '
                final_sentences.append(sentence)

        return final_sentences

    def _get_overlap_text(self, text: str) -> str:
        """
        Get the last portion of text for overlap with next chunk.

        Args:
            text: Full text of current chunk

        Returns:
            Overlap text for next chunk
        """
        if len(text) <= self.overlap:
            return text

        # Try to find a good breaking point for overlap (sentence boundary)
        overlap_start = len(text) - self.overlap
        overlap_text = text[overlap_start:]

        # Try to start overlap at a sentence boundary
        sentence_start = overlap_text.find('. ')
        if sentence_start != -1 and sentence_start < self.overlap // 2:
            overlap_text = overlap_text[sentence_start + 2:]

        return overlap_text

    def _create_chunk(self, document: Dict, content: str, chunk_index: int) -> Dict:
        """
        Create a chunk dictionary.

        Args:
            document: Source document
            content: Chunk content
            chunk_index: Index of this chunk within the document

        Returns:
            Chunk dictionary
        """
        return {
            'chunk_id': f"{document['filename']}_chunk_{chunk_index}",
            'document_name': document['filename'],
            'chunk_index': chunk_index,
            'content': content.strip(),
            'char_count': len(content.strip()),
            'word_count': len(content.strip().split()),
            'source_file': document['file_path']
        }

    def get_chunk_stats(self) -> Dict:
        """
        Get statistics about the created chunks.

        Returns:
            Dictionary with chunk statistics
        """
        if not self.chunks:
            return {"error": "No chunks created"}

        total_words = sum(chunk['word_count'] for chunk in self.chunks)
        total_chars = sum(chunk['char_count'] for chunk in self.chunks)
        avg_words_per_chunk = total_words / len(self.chunks)
        avg_chars_per_chunk = total_chars / len(self.chunks)

        # Find documents with most chunks
        doc_chunk_counts = {}
        for chunk in self.chunks:
            doc_name = chunk['document_name']
            doc_chunk_counts[doc_name] = doc_chunk_counts.get(doc_name, 0) + 1

        most_chunked_doc = max(doc_chunk_counts.items(), key=lambda x: x[1])

        return {
            'total_chunks': len(self.chunks),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_chunk': round(avg_words_per_chunk, 2),
            'average_chars_per_chunk': round(avg_chars_per_chunk, 2),
            'largest_chunk': max(self.chunks, key=lambda x: x['word_count']),
            'smallest_chunk': min(self.chunks, key=lambda x: x['word_count']),
            'most_chunked_document': most_chunked_doc,
            'documents_chunked': len(set(chunk['document_name'] for chunk in self.chunks))
        }

    def save_chunks_sample(self, num_samples: int = 5) -> None:
        """
        Display sample chunks for review.

        Args:
            num_samples: Number of sample chunks to display
        """
        if not self.chunks:
            print("No chunks to display")
            return

        print(f"\n=== Sample of {min(num_samples, len(self.chunks))} Chunks ===")

        for i in range(min(num_samples, len(self.chunks))):
            chunk = self.chunks[i]
            print(f"\nChunk {i + 1}:")
            print(f"ID: {chunk['chunk_id']}")
            print(f"Source: {chunk['document_name']}")
            print(f"Words: {chunk['word_count']}, Characters: {chunk['char_count']}")
            print(f"Content preview: {chunk['content'][:200]}...")
            print("-" * 50)


# Example usage and testing
if __name__ == "__main__":
    # Load documents first
    folder_path = r"C:\Users\Yesh\PycharmProjects\insurance-rag-system\data\text_files"
    loader = DocumentLoader(folder_path)

    try:
        print("Loading documents...")
        documents = loader.load_all_documents()

        # Create chunker and process documents
        print("\nCreating chunks...")
        chunker = TextChunker(chunk_size=1000, overlap=200)
        chunks = chunker.chunk_documents(documents)

        # Show statistics
        stats = chunker.get_chunk_stats()
        print("\n=== Chunk Statistics ===")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Average words per chunk: {stats['average_words_per_chunk']}")
        print(f"Average characters per chunk: {stats['average_chars_per_chunk']}")
        print(
            f"Most chunked document: {stats['most_chunked_document'][0]} ({stats['most_chunked_document'][1]} chunks)")
        print(f"Largest chunk: {stats['largest_chunk']['chunk_id']} ({stats['largest_chunk']['word_count']} words)")
        print(f"Smallest chunk: {stats['smallest_chunk']['chunk_id']} ({stats['smallest_chunk']['word_count']} words)")

        # Show sample chunks
        chunker.save_chunks_sample(3)

    except Exception as e:
        print(f"Error: {str(e)}")