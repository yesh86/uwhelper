"""
Smart Sentence-Based Text Chunker for Insurance Documents
Preserves sentence boundaries and semantic context
"""

import re
import nltk
from typing import List, Dict, Tuple
import os
nltk.download()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK sentence tokenizer...")
    nltk.download('punkt')


class SmartSentenceChunker:
    def __init__(self,
                 sentences_per_chunk: int = 5,  # Fixed number of sentences per chunk
                 overlap_sentences: int = 2,  # Sentences to overlap between chunks
                 min_sentence_length: int = 20,  # Minimum sentence length to include
                 max_chunk_chars: int = 2000):  # Safety limit for very long sentences
        """
        Initialize the smart sentence chunker with fixed sentence counts.

        Args:
            sentences_per_chunk: Number of sentences per chunk
            overlap_sentences: Number of sentences to overlap between chunks
            min_sentence_length: Minimum length for a sentence to be included
            max_chunk_chars: Maximum characters per chunk (safety limit)
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.min_sentence_length = min_sentence_length
        self.max_chunk_chars = max_chunk_chars

        # Validate parameters
        if overlap_sentences >= sentences_per_chunk:
            raise ValueError("Overlap sentences must be less than sentences per chunk")

        print(f"📝 Chunker config: {sentences_per_chunk} sentences per chunk, {overlap_sentences} sentence overlap")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better sentence detection."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common sentence boundary issues in insurance docs
        # Fix numbered lists that might confuse sentence tokenizer
        text = re.sub(r'(\d+)\.\s*([A-Z])', r'\1. \2', text)

        # Fix abbreviations that might cause false sentence breaks
        abbreviations = ['Inc', 'Corp', 'Ltd', 'Co', 'LLC', 'Mr', 'Mrs', 'Dr', 'Prof', 'vs', 'etc']
        for abbr in abbreviations:
            text = re.sub(f'{abbr}\.', f'{abbr}[DOT]', text)

        # Fix decimal numbers
        text = re.sub(r'(\d+)\.(\d+)', r'\1[DOT]\2', text)

        return text.strip()

    def _restore_dots(self, text: str) -> str:
        """Restore dots that were temporarily replaced."""
        return text.replace('[DOT]', '.')

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK with insurance document awareness."""
        cleaned_text = self._clean_text(text)

        # Use NLTK sentence tokenizer
        sentences = nltk.sent_tokenize(cleaned_text)

        # Restore dots and clean up sentences
        sentences = [self._restore_dots(sent.strip()) for sent in sentences]

        # Filter out very short sentences (likely fragments)
        sentences = [sent for sent in sentences if len(sent) > 20]

        return sentences

    def _combine_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Combine sentences into fixed-size chunks with overlap."""
        if not sentences:
            return []

        # Filter sentences by minimum length
        valid_sentences = [s for s in sentences if len(s) >= self.min_sentence_length]

        if len(valid_sentences) < self.sentences_per_chunk:
            # If we don't have enough sentences, create one chunk with all sentences
            if valid_sentences:
                return [' '.join(valid_sentences)]
            else:
                return []

        chunks = []
        start_idx = 0

        while start_idx < len(valid_sentences):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.sentences_per_chunk, len(valid_sentences))

            # Get sentences for this chunk
            chunk_sentences = valid_sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)

            # Check if chunk exceeds character limit (safety check)
            if len(chunk_text) > self.max_chunk_chars:
                print(f"⚠️ Warning: Chunk exceeds {self.max_chunk_chars} characters ({len(chunk_text)})")
                # Still include it but note the warning

            chunks.append(chunk_text)

            # Calculate next start position with overlap
            # If this is the last possible chunk, break
            if end_idx >= len(valid_sentences):
                break

            # Move forward by (sentences_per_chunk - overlap_sentences)
            start_idx += (self.sentences_per_chunk - self.overlap_sentences)

            # Ensure we don't create duplicate chunks
            if start_idx >= len(valid_sentences) - self.overlap_sentences:
                break

        return chunks

    def chunk_text(self, text: str, source_info: Dict = None) -> List[Dict]:
        """
        Chunk text into sentence-based chunks with metadata.

        Args:
            text: Input text to chunk
            source_info: Optional metadata about the source

        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text or len(text.strip()) < 50:
            return []

        # Split into sentences
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # Combine sentences into chunks
        chunks = self._combine_sentences_into_chunks(sentences)

        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_obj = {
                'content': chunk_text,
                'chunk_id': i,
                'chunk_length': len(chunk_text),
                'sentence_count': len([s for s in sentences if s in chunk_text]),
                'source_info': source_info or {},
            }
            chunk_objects.append(chunk_obj)

        return chunk_objects

    def chunk_document(self, file_path: str) -> List[Dict]:
        """
        Load and chunk a document file.

        Args:
            file_path: Path to the document file

        Returns:
            List of chunk dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            source_info = {
                'document_name': os.path.basename(file_path),
                'file_path': file_path,
                'document_length': len(content)
            }

            return self.chunk_text(content, source_info)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def get_chunking_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about the chunking results."""
        if not chunks:
            return {}

        chunk_lengths = [chunk['chunk_length'] for chunk in chunks]
        sentence_counts = [chunk['sentence_count'] for chunk in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunks),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'avg_sentences_per_chunk': sum(sentence_counts) / len(chunks),
            'total_content_length': sum(chunk_lengths),
            'sentences_per_chunk_target': self.sentences_per_chunk,
            'overlap_sentences': self.overlap_sentences
        }


def test_sentence_chunker():
    """Test the sentence chunker with sample insurance text."""

    sample_text = """
    Equipment Breakdown Protection Coverage provides coverage for direct physical loss or damage to Covered Equipment caused by or resulting from a Breakdown. This coverage applies to equipment that is owned by you or leased by you under a written agreement. The coverage is subject to all terms, conditions, and exclusions of this policy.

    Covered Equipment means equipment that generates, transmits or utilizes energy for its operation. It also includes equipment that has moving parts or components that move or rotate during normal operation. Additionally, it covers equipment that operates under vacuum or pressure conditions. Finally, it includes equipment that is connected to a computer system or network for monitoring or control purposes.

    Breakdown means sudden and accidental physical damage to Covered Equipment that requires repair or replacement of the Covered Equipment or a part of the Covered Equipment. The damage must be due to causes other than fire, lightning, windstorm, flood, earthquake, theft, or other causes of loss covered under other insurance. The breakdown must manifest itself at the time of its occurrence by physical damage to the Covered Equipment that necessitates repair or replacement.

    This coverage shall not apply to loss or damage caused by wear and tear, gradual deterioration, or any other gradually developing condition. It also excludes damage from rust, corrosion, fungus, decay, or any condition due to gradual deterioration over time. Coverage does not extend to damage caused by insects or vermin that may affect the equipment. Finally, inherent vice, latent defect, or any quality in property that causes it to damage or destroy itself is not covered.

    The limit of liability for this coverage is the amount shown in the Declarations for Equipment Breakdown Protection. This limit applies regardless of the number of items of Covered Equipment involved in any one occurrence. The deductible amount shown in the Declarations applies to each occurrence and will be deducted from the amount of loss or damage before any payment is made under this coverage.
    """

    # Test with 5 sentences per chunk, 2 sentence overlap
    chunker = SmartSentenceChunker(sentences_per_chunk=5, overlap_sentences=2)
    chunks = chunker.chunk_text(sample_text)

    print("🧪 Testing Fixed Sentence-Based Chunker")
    print("📝 Configuration: 5 sentences per chunk, 2 sentence overlap")
    print("=" * 70)

    for i, chunk in enumerate(chunks):
        print(f"\n📄 Chunk {i + 1}:")
        print(f"   Length: {chunk['chunk_length']} characters")
        print(f"   Sentences: {chunk['sentence_count']}")
        print(f"   Content preview: {chunk['content'][:150]}...")

        # Show the actual sentences in this chunk
        sentences = chunk['content'].split('. ')
        print(f"   Sentences in chunk:")
        for j, sent in enumerate(sentences[:3]):  # Show first 3 sentences
            print(f"     {j + 1}. {sent[:80]}...")

    stats = chunker.get_chunking_stats(chunks)
    print(f"\n📊 Chunking Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Show overlap demonstration
    if len(chunks) > 1:
        print(f"\n🔗 Overlap Demonstration (Chunk 1 vs Chunk 2):")
        chunk1_sentences = chunks[0]['content'].split('. ')
        chunk2_sentences = chunks[1]['content'].split('. ')

        print(f"   Last 2 sentences of Chunk 1:")
        for sent in chunk1_sentences[-2:]:
            print(f"     • {sent[:100]}...")

        print(f"   First 2 sentences of Chunk 2:")
        for sent in chunk2_sentences[:2]:
            print(f"     • {sent[:100]}...")


if __name__ == "__main__":
    test_sentence_chunker()