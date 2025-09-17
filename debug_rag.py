import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import json


class EnhancedRAGPipeline:
    def __init__(self, chunks, embeddings, embedding_function, anthropic_client):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_function = embedding_function
        self.anthropic_client = anthropic_client
        self.debug_mode = True  # Set to False in production

    def retrieve_with_debug(self, query, top_k=5, similarity_threshold=0.3):
        """
        Enhanced retrieval with debugging information
        """
        # Generate query embedding
        query_embedding = self.embedding_function(query)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get all results sorted by similarity
        all_indices = np.argsort(similarities)[::-1]
        all_scores = similarities[all_indices]

        if self.debug_mode:
            print(f"\n🔍 DEBUG: Query retrieval for '{query}'")
            print(f"📊 Similarity scores - Max: {all_scores[0]:.4f}, Min: {all_scores[-1]:.4f}")
            print(f"🎯 Threshold: {similarity_threshold}, Top-K: {top_k}")

            # Check if form number exists in any chunk
            form_found_indices = []
            for i, chunk in enumerate(self.chunks):
                if "AG 99 07 12 19" in str(chunk).upper():
                    form_found_indices.append(i)
                    score = similarities[i]
                    rank = np.where(all_indices == i)[0][0] + 1
                    print(f"📍 FORM FOUND in chunk {i}: Score={score:.4f}, Rank={rank}")

                    # Show why it might not be retrieved
                    if score < similarity_threshold:
                        print(f"   ⚠️  BELOW THRESHOLD! Score {score:.4f} < {similarity_threshold}")
                    if rank > top_k:
                        print(f"   ⚠️  BELOW TOP-K! Rank {rank} > {top_k}")

            if not form_found_indices:
                print("❌ Form number 'AG 99 07 12 19' not found in any chunk")

            # Show top retrieved chunks
            print(f"\n📋 TOP {min(top_k, len(all_indices))} RETRIEVED CHUNKS:")
            for i in range(min(top_k, len(all_indices))):
                idx = all_indices[i]
                score = all_scores[i]
                chunk_preview = str(self.chunks[idx])[:100] + "..."
                contains_form = "🎯" if "AG 99 07 12 19" in str(self.chunks[idx]).upper() else "  "
                print(f"  {i + 1}. {contains_form} Chunk {idx} (Score: {score:.4f}): {chunk_preview}")

        # Apply filtering
        top_indices = all_indices[:top_k]
        top_scores = all_scores[:top_k]

        # Apply threshold
        filtered_results = [(idx, score) for idx, score in zip(top_indices, top_scores)
                            if score >= similarity_threshold]

        retrieved_chunks = [self.chunks[idx] for idx, _ in filtered_results]

        if self.debug_mode:
            print(f"\n✅ FINAL RETRIEVAL: {len(retrieved_chunks)} chunks passed filters")
            form_in_retrieved = any("AG 99 07 12 19" in str(chunk).upper()
                                    for chunk in retrieved_chunks)
            print(f"🎯 Form number in retrieved chunks: {'✅ YES' if form_in_retrieved else '❌ NO'}")

        return retrieved_chunks, filtered_results

    def query_with_debug(self, user_query, top_k=5, similarity_threshold=0.3):
        """
        Complete RAG query with debugging
        """
        print(f"\n" + "=" * 60)
        print(f"🚀 RAG QUERY: {user_query}")
        print("=" * 60)

        # Retrieve relevant chunks
        retrieved_chunks, chunk_scores = self.retrieve_with_debug(
            user_query, top_k, similarity_threshold
        )

        if not retrieved_chunks:
            print("❌ No chunks retrieved - adjusting parameters...")
            # Try with lower threshold and higher top_k
            retrieved_chunks, chunk_scores = self.retrieve_with_debug(
                user_query, top_k=10, similarity_threshold=0.1
            )

        # Prepare context for Anthropic
        context = "\n\n---\n\n".join([f"Chunk {i + 1}:\n{chunk}"
                                      for i, chunk in enumerate(retrieved_chunks)])

        # Create the prompt
        prompt = f"""Based on the provided documentation, please answer the following question:

Question: {user_query}

Documentation:
{context}

Please provide a comprehensive answer based on the provided information. If the specific information is not available in the provided documentation, please state that clearly."""

        # Query Anthropic
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.content[0].text

            if self.debug_mode:
                print(f"\n📤 ANTHROPIC RESPONSE:")
                print(f"Response length: {len(answer)} characters")
                form_mentioned = "AG 99 07 12 19" in answer.upper()
                print(f"Form number mentioned in response: {'✅ YES' if form_mentioned else '❌ NO'}")

            return {
                'answer': answer,
                'retrieved_chunks': len(retrieved_chunks),
                'chunk_scores': chunk_scores,
                'context_sent': len(context),
                'debug_info': {
                    'form_in_chunks': any("AG 99 07 12 19" in str(chunk).upper()
                                          for chunk in retrieved_chunks),
                    'total_chunks_searched': len(self.chunks)
                }
            }

        except Exception as e:
            print(f"❌ Error querying Anthropic: {e}")
            return None


def fix_rag_parameters(rag_pipeline):
    """
    Suggest optimal parameters based on the data
    """
    print("\n🔧 PARAMETER OPTIMIZATION SUGGESTIONS:")

    # Test different thresholds
    test_query = "AG 99 07 12 19"
    query_embedding = rag_pipeline.embedding_function(test_query)
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    similarities = cosine_similarity(query_embedding, rag_pipeline.embeddings)[0]

    # Find chunks with the form number
    form_chunks = []
    for i, chunk in enumerate(rag_pipeline.chunks):
        if "AG 99 07 12 19" in str(chunk).upper():
            form_chunks.append((i, similarities[i]))

    if form_chunks:
        min_score = min(score for _, score in form_chunks)
        max_score = max(score for _, score in form_chunks)

        print(f"📊 Form number similarity scores: {min_score:.4f} - {max_score:.4f}")
        print(f"💡 Recommended threshold: {min_score - 0.05:.4f} (to capture all form instances)")

        # Find rank of worst form chunk
        worst_rank = len(similarities) - np.searchsorted(np.sort(similarities), min_score)
        print(f"💡 Recommended top_k: {worst_rank + 2} (to capture all form instances)")

    return form_chunks


# Example usage - integrate this into your app.py
def main():
    # Your existing setup
    # chunks = load_your_chunks()
    # embeddings = load_your_embeddings()
    # embedding_function = your_embedding_function
    # anthropic_client = anthropic.Anthropic(api_key="your-key")

    # Create enhanced pipeline
    # rag = EnhancedRAGPipeline(chunks, embeddings, embedding_function, anthropic_client)

    # Test with the problematic query
    # result = rag.query_with_debug("What is AG 99 07 12 19?")

    # Get parameter suggestions
    # fix_rag_parameters(rag)

    pass


if __name__ == "__main__":
    main()