"""
Complete Insurance Document Processing Pipeline
Processes documents from data/text_files with smart sentence chunking
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import glob
from tqdm import tqdm
from sentence_chunker import SmartSentenceChunker
from typing import List, Dict

class InsuranceDocumentProcessor:
    def __init__(self,
                 documents_dir: str = "data/text_files",
                 embeddings_dir: str = "embeddings",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document processor.

        Args:
            documents_dir: Directory containing text files
            embeddings_dir: Directory to save embeddings
            model_name: Sentence transformer model name
        """
        self.documents_dir = documents_dir
        self.embeddings_dir = embeddings_dir
        self.model_name = model_name

        # Initialize components
        self.chunker = SmartSentenceChunker(
            sentences_per_chunk=5,    # 5 sentences per chunk as requested
            overlap_sentences=2,      # 2 sentence overlap as requested
            min_sentence_length=20,   # Minimum sentence length
            max_chunk_chars=2500      # Safety limit for very long sentences
        )
        self.embedding_model = None

        # Data storage
        self.all_chunks = []
        self.all_metadata = []

    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        print(f"📥 Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        print("✅ Embedding model loaded successfully!")

    def _find_text_files(self) -> List[str]:
        """Find all text files in the documents directory."""
        patterns = [
            os.path.join(self.documents_dir, "*.txt"),
            os.path.join(self.documents_dir, "*.TXT"),
        ]

        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))

        return sorted(files)

    def _process_documents(self):
        """Process all documents and create chunks."""
        text_files = self._find_text_files()

        if not text_files:
            raise ValueError(f"No text files found in {self.documents_dir}")

        print(f"📚 Found {len(text_files)} text files to process")

        total_chunks = 0

        for file_path in tqdm(text_files, desc="Processing documents"):
            try:
                print(f"\n📄 Processing: {os.path.basename(file_path)}")

                # Chunk the document
                chunks = self.chunker.chunk_document(file_path)

                if chunks:
                    self.all_chunks.extend([chunk['content'] for chunk in chunks])

                    # Create metadata for each chunk
                    for chunk in chunks:
                        metadata = {
                            'document_name': chunk['source_info']['document_name'],
                            'file_path': chunk['source_info']['file_path'],
                            'chunk_id': chunk['chunk_id'],
                            'chunk_length': chunk['chunk_length'],
                            'sentence_count': chunk['sentence_count'],
                            'global_chunk_id': len(self.all_metadata)
                        }
                        self.all_metadata.append(metadata)

                    total_chunks += len(chunks)
                    print(f"   ✅ Created {len(chunks)} chunks")

                    # Show chunking stats for this document
                    stats = self.chunker.get_chunking_stats(chunks)
                    print(f"   📊 Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
                    print(f"   📊 Avg sentences per chunk: {stats['avg_sentences_per_chunk']:.1f}")
                else:
                    print(f"   ⚠️ No chunks created (file too short or empty)")

            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {str(e)}")
                continue

        print(f"\n📊 Total chunks created: {total_chunks}")
        print(f"📊 Average chunk length: {np.mean([len(chunk) for chunk in self.all_chunks]):.0f} characters")

    def _generate_embeddings(self):
        """Generate embeddings for all chunks."""
        if not self.all_chunks:
            raise ValueError("No chunks to embed. Process documents first.")

        print(f"🔄 Generating embeddings for {len(self.all_chunks)} chunks...")

        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []

        for i in tqdm(range(0, len(self.all_chunks), batch_size), desc="Creating embeddings"):
            batch = self.all_chunks[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            all_embeddings.append(embeddings)

        # Combine all embeddings
        embeddings_matrix = np.vstack(all_embeddings)

        print(f"✅ Generated embeddings matrix: {embeddings_matrix.shape}")
        return embeddings_matrix

    def _save_everything(self, embeddings_matrix):
        """Save all processed data."""
        print(f"💾 Saving data to {self.embeddings_dir}...")

        # Create embeddings directory
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Save embeddings
        embeddings_file = os.path.join(self.embeddings_dir, "embeddings.npy")
        np.save(embeddings_file, embeddings_matrix)
        print(f"✅ Embeddings saved: {embeddings_file}")

        # Save chunks metadata
        metadata_file = os.path.join(self.embeddings_dir, "chunks_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.all_metadata, f)
        print(f"✅ Metadata saved: {metadata_file}")

        # Save actual chunk texts separately for easy retrieval
        chunks_file = os.path.join(self.embeddings_dir, "chunks.pkl")
        with open(chunks_file, 'wb') as f:
            pickle.dump(self.all_chunks, f)
        print(f"✅ Chunk texts saved: {chunks_file}")

        # Save configuration
        config = {
            'model_name': self.model_name,
            'total_chunks': len(self.all_chunks),
            'total_documents': len(set([meta['document_name'] for meta in self.all_metadata])),
            'chunking_method': 'fixed_sentence_based',
            'sentences_per_chunk': self.chunker.sentences_per_chunk,
            'overlap_sentences': self.chunker.overlap_sentences,
            'embedding_dimensions': embeddings_matrix.shape[1]
        }

        config_file = os.path.join(self.embeddings_dir, "config.pkl")
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        print(f"✅ Config saved: {config_file}")

        return config

    def process_all(self):
        """Run the complete processing pipeline."""
        print("🚀 Starting Insurance Document Processing Pipeline")
        print("=" * 60)

        try:
            # Step 1: Load embedding model
            self._load_embedding_model()

            # Step 2: Process documents and create chunks
            self._process_documents()

            if not self.all_chunks:
                print("❌ No chunks were created. Check your documents directory.")
                return False

            # Step 3: Generate embeddings
            embeddings_matrix = self._generate_embeddings()

            # Step 4: Save everything
            config = self._save_everything(embeddings_matrix)

            # Print summary
            print("\n" + "=" * 60)
            print("🎉 PROCESSING COMPLETE!")
            print("=" * 60)
            print(f"📚 Documents processed: {config['total_documents']}")
            print(f"📄 Total chunks: {config['total_chunks']}")
            print(f"🔍 Embedding model: {config['model_name']}")
            print(f"📏 Embedding dimensions: {config['embedding_dimensions']}")
            print(f"✂️ Chunking method: {config['chunking_method']}")
            print(f"🎯 Target chunk size: {config['target_chunk_size']} characters")
            print("\nYou can now run:")
            print("  python claude_insurance_rag.py")
            print("  python app.py")

            return True

        except Exception as e:
            print(f"❌ Error in processing pipeline: {str(e)}")
            return False


def main():
    # Check if documents directory exists
    docs_dir = "data/text_files"
    if not os.path.exists(docs_dir):
        print(f"❌ Documents directory '{docs_dir}' not found!")
        print("Please create the directory and add your .txt files")

        # Try alternative directories
        alt_dirs = ["data", "documents", "text_files"]
        for alt_dir in alt_dirs:
            if os.path.exists(alt_dir):
                txt_files = glob.glob(os.path.join(alt_dir, "*.txt"))
                if txt_files:
                    print(f"\n💡 Found .txt files in '{alt_dir}':")
                    for file in txt_files[:5]:
                        print(f"   - {os.path.basename(file)}")

                    use_alt = input(f"\nUse '{alt_dir}' as documents directory? (y/n): ")
                    if use_alt.lower() == 'y':
                        docs_dir = alt_dir
                        break
        else:
            return

    # Initialize and run processor
    processor = InsuranceDocumentProcessor(documents_dir=docs_dir)
    success = processor.process_all()

    if success:
        print("\n🎯 Next steps:")
        print("1. Test your RAG system: python claude_insurance_rag.py")
        print("2. Run the web interface: python app.py")
    else:
        print("\n❌ Processing failed. Check the errors above.")


if __name__ == "__main__":
    main()