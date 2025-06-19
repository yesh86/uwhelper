"""
Embeddings Generator for Insurance RAG System
This file handles creating vector embeddings from text chunks for semantic search.
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm

from document_loader import DocumentLoader
from text_chunker import TextChunker


class EmbeddingsGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings generator.

        Args:
            model_name: Name of the sentence transformer model to use
                       "all-MiniLM-L6-v2" is fast and good for general use
                       "all-mpnet-base-v2" is more accurate but slower
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        print("This might take a minute on first run...")

        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        self.embeddings = None
        self.chunks_with_embeddings = []

    def generate_embeddings(self, chunks: List[Dict], batch_size: int = 32, save_path: str = None) -> List[Dict]:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of chunk dictionaries from TextChunker
            batch_size: Number of chunks to process at once (adjust based on your RAM)
            save_path: Optional path to save embeddings (recommended for large datasets)

        Returns:
            List of chunks with embeddings added
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")
        print(f"Using batch size: {batch_size}")

        # Extract text content from chunks
        chunk_texts = [chunk['content'] for chunk in chunks]

        # Generate embeddings in batches with progress bar
        start_time = time.time()

        print("Creating embeddings...")
        embeddings = self.model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Add embeddings to chunk dictionaries
        chunks_with_embeddings = []
        for i, chunk in enumerate(chunks):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embeddings[i]
            chunk_with_embedding['embedding_model'] = self.model_name
            chunks_with_embeddings.append(chunk_with_embedding)

        self.chunks_with_embeddings = chunks_with_embeddings
        self.embeddings = embeddings

        elapsed_time = time.time() - start_time
        print(f"Embedding generation completed in {elapsed_time:.2f} seconds")
        print(f"Average time per chunk: {elapsed_time / len(chunks):.4f} seconds")

        # Save embeddings if path provided
        if save_path:
            self.save_embeddings(save_path)

        return chunks_with_embeddings

    def save_embeddings(self, save_path: str) -> None:
        """
        Save embeddings and chunk data to disk.

        Args:
            save_path: Directory path to save the files
        """
        if not self.chunks_with_embeddings:
            print("No embeddings to save. Generate embeddings first.")
            return

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save embeddings as numpy array
        embeddings_file = os.path.join(save_path, "embeddings.npy")
        np.save(embeddings_file, self.embeddings)

        # Save chunks with metadata (without embeddings to save space)
        chunks_metadata = []
        for chunk in self.chunks_with_embeddings:
            metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
            chunks_metadata.append(metadata)

        metadata_file = os.path.join(save_path, "chunks_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(chunks_metadata, f)

        # Save configuration
        config = {
            'model_name': self.model_name,
            'total_chunks': len(self.chunks_with_embeddings),
            'embedding_dimension': self.embeddings.shape[1],
            'creation_time': time.time()
        }

        config_file = os.path.join(save_path, "config.pkl")
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)

        print(f"Embeddings saved to: {save_path}")
        print(f"Files created:")
        print(f"  - embeddings.npy ({self.embeddings.nbytes / 1024 / 1024:.2f} MB)")
        print(f"  - chunks_metadata.pkl")
        print(f"  - config.pkl")

    def load_embeddings(self, save_path: str) -> bool:
        """
        Load previously saved embeddings.

        Args:
            save_path: Directory path where embeddings are saved

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load embeddings
            embeddings_file = os.path.join(save_path, "embeddings.npy")
            self.embeddings = np.load(embeddings_file)

            # Load metadata
            metadata_file = os.path.join(save_path, "chunks_metadata.pkl")
            with open(metadata_file, 'rb') as f:
                chunks_metadata = pickle.load(f)

            # Load config
            config_file = os.path.join(save_path, "config.pkl")
            with open(config_file, 'rb') as f:
                config = pickle.load(f)

            # Reconstruct chunks with embeddings
            self.chunks_with_embeddings = []
            for i, metadata in enumerate(chunks_metadata):
                chunk_with_embedding = metadata.copy()
                chunk_with_embedding['embedding'] = self.embeddings[i]
                chunk_with_embedding['embedding_model'] = config['model_name']
                self.chunks_with_embeddings.append(chunk_with_embedding)

            self.model_name = config['model_name']

            print(f"Successfully loaded {len(self.chunks_with_embeddings)} embeddings")
            print(f"Model: {config['model_name']}")
            print(f"Embedding dimension: {config['embedding_dimension']}")

            return True

        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return False

    def get_embedding_stats(self) -> Dict:
        """
        Get statistics about the embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        if not self.embeddings is not None:
            return {"error": "No embeddings generated"}

        return {
            'total_embeddings': len(self.embeddings),
            'embedding_dimension': self.embeddings.shape[1],
            'model_name': self.model_name,
            'memory_usage_mb': self.embeddings.nbytes / 1024 / 1024,
            'average_embedding_norm': np.mean(np.linalg.norm(self.embeddings, axis=1)),
            'embedding_shape': self.embeddings.shape
        }

    def find_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find chunks most similar to a query (basic similarity search).

        Args:
            query: Text query to search for
            top_k: Number of most similar chunks to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.chunks_with_embeddings:
            print("No embeddings available. Generate embeddings first.")
            return []

        # Generate embedding for query
        query_embedding = self.model.encode([query])

        # Calculate cosine similarity with all chunk embeddings
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        similarities = similarities / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))

        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            chunk = self.chunks_with_embeddings[idx]
            score = float(similarities[idx])
            results.append((chunk, score))

        return results


# Example usage and testing
if __name__ == "__main__":
    # Load documents and create chunks
    folder_path = r"C:\Users\Yesh\PycharmProjects\insurance-rag-system\data\text_files"
    save_embeddings_path = r"C:\Users\Yesh\PycharmProjects\insurance-rag-system\embeddings"

    try:
        print("Step 1: Loading documents...")
        loader = DocumentLoader(folder_path)
        documents = loader.load_all_documents()

        print("\nStep 2: Creating chunks...")
        chunker = TextChunker(chunk_size=1000, overlap=200)
        chunks = chunker.chunk_documents(documents)

        print(f"\nStep 3: Generating embeddings for {len(chunks)} chunks...")
        embedder = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")

        # For 44k chunks, this might take 10-20 minutes
        # You can reduce batch_size if you get memory errors
        chunks_with_embeddings = embedder.generate_embeddings(
            chunks,
            batch_size=32,
            save_path=save_embeddings_path
        )

        # Show statistics
        stats = embedder.get_embedding_stats()
        print("\n=== Embedding Statistics ===")
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
        print(f"Model: {stats['model_name']}")

        # Test similarity search
        print("\n=== Testing Similarity Search ===")
        test_query = "what is covered under liability insurance"
        similar_chunks = embedder.find_similar_chunks(test_query, top_k=3)

        print(f"Query: '{test_query}'")
        print("Most similar chunks:")
        for i, (chunk, score) in enumerate(similar_chunks, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   Source: {chunk['document_name']}")
            print(f"   Content: {chunk['content'][:200]}...")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have sentence-transformers installed:")
        print("pip install sentence-transformers")