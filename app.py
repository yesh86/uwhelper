import os

# Set API key FIRST, before any other imports or initialization
os.environ['ANTHROPIC_API_KEY'] = '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'
print(f"🔑 API key set: {os.environ.get('ANTHROPIC_API_KEY', 'NOT FOUND')[:20]}...")

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import re
from datetime import datetime
import logging

# Load environment variables (but API key is already set above)
load_dotenv()

# Test API key immediately after setting it
print("🧪 Testing API key before RAG initialization...")
try:
    test_client = Anthropic()
    test_response = test_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=5,
        messages=[{"role": "user", "content": "Hi"}]
    )
    print("✅ API key works before RAG init!")
except Exception as e:
    print(f"❌ API key failed before RAG init: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaudeInsuranceRAG:
    def __init__(self, embeddings_dir="embeddings", model_name="all-MiniLM-L6-v2"):
        """Initialize the Claude-powered Insurance RAG system"""
        self.embeddings_dir = embeddings_dir
        self.model_name = model_name
        self.embedding_model = None
        self.embeddings = None
        self.chunks_metadata = []
        self.config = {}
        self.anthropic_client = None

        # Load the system
        self._load_system()

    def _load_system(self):
        """Load all components of the RAG system"""
        try:
            logger.info("🚀 Loading Claude Enhanced Insurance RAG System...")

            # Load embeddings (try both with and without .npy extension)
            embeddings_files = [
                os.path.join(self.embeddings_dir, "embeddings.npy"),
                os.path.join(self.embeddings_dir, "embeddings")
            ]

            embeddings_loaded = False
            for embeddings_file in embeddings_files:
                if os.path.exists(embeddings_file):
                    self.embeddings = np.load(embeddings_file)
                    logger.info(f"✅ Loaded embeddings: {self.embeddings.shape}")
                    embeddings_loaded = True
                    break

            if not embeddings_loaded:
                raise FileNotFoundError(f"Embeddings not found. Tried: {embeddings_files}")

            # Load chunks metadata (try both with and without .pkl extension)
            metadata_files = [
                os.path.join(self.embeddings_dir, "chunks_metadata.pkl"),
                os.path.join(self.embeddings_dir, "chunks_metadata")
            ]

            metadata_loaded = False
            for metadata_file in metadata_files:
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'rb') as f:
                        self.chunks_metadata = pickle.load(f)
                    logger.info(f"✅ Loaded {len(self.chunks_metadata)} chunk metadata entries")
                    metadata_loaded = True
                    break

            if not metadata_loaded:
                raise FileNotFoundError(f"Metadata not found. Tried: {metadata_files}")

            # Load config (try both with and without .pkl extension)
            config_files = [
                os.path.join(self.embeddings_dir, "config.pkl"),
                os.path.join(self.embeddings_dir, "config")
            ]

            config_loaded = False
            for config_file in config_files:
                if os.path.exists(config_file):
                    with open(config_file, 'rb') as f:
                        self.config = pickle.load(f)
                    logger.info(f"✅ Loaded config: {self.config.get('chunking_method', 'unknown')} chunking")
                    config_loaded = True
                    break

            if not config_loaded:
                raise FileNotFoundError(f"Config not found. Tried: {config_files}")

            # Load embedding model
            model_name = self.config.get('model_name', self.model_name)
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)

            # Initialize Anthropic client with error handling for different versions
            try:
                # Set API key again just before creating client
                api_key = os.environ.get('ANTHROPIC_API_KEY')
                if not api_key or not api_key.startswith('sk-ant-'):
                    raise ValueError("API key not found or invalid format")

                logger.info(f"Creating Anthropic client with key: {api_key[:20]}...")
                self.anthropic_client = Anthropic(api_key=api_key)  # Pass explicitly
                logger.info("✅ Anthropic client initialized successfully")
            except TypeError as e:
                if "proxies" in str(e):
                    # Handle version compatibility issue
                    logger.warning("Anthropic client version compatibility issue, trying alternative initialization...")
                    try:
                        # Try importing the client differently for older versions
                        import anthropic
                        self.anthropic_client = anthropic.Client()
                        logger.info("✅ Anthropic client initialized with alternative method")
                    except:
                        # If all else fails, initialize without optional parameters
                        self.anthropic_client = Anthropic()
                        logger.info("✅ Anthropic client initialized with basic initialization")
                else:
                    raise

            logger.info(f"✅ Claude Enhanced system loaded successfully!")
            logger.info(f"📚 Knowledge base: {self.config.get('total_chunks', len(self.chunks_metadata))} chunks")
            logger.info(f"🔍 Embedding model: {model_name}")
            logger.info(f"🤖 Response generation: Claude 3.5 Sonnet")

        except Exception as e:
            logger.error(f"❌ Error loading RAG system: {str(e)}")
            raise

    def _detect_query_type(self, query):
        """Detect the type of query and extract relevant information"""
        query_lower = query.lower()

        # Form code pattern (e.g., BM_00_20_07_01)
        form_pattern = r'([A-Z]{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})'
        form_match = re.search(form_pattern, query.replace('-', '_').upper())

        if form_match:
            return {
                'type': 'form_specific',
                'form_code': form_match.group(1),
                'subtype': 'form_overview' if any(
                    word in query_lower for word in ['what is', 'about', 'explain', 'describe']) else 'form_detail'
            }
        elif any(word in query_lower for word in ['coverage', 'covers', 'covered', 'protection']):
            return {'type': 'coverage_inquiry'}
        elif any(word in query_lower for word in ['claim', 'claims', 'file a claim']):
            return {'type': 'claims_process'}
        elif any(word in query_lower for word in ['premium', 'cost', 'price', 'rate']):
            return {'type': 'pricing_inquiry'}
        else:
            return {'type': 'general_inquiry'}

    def _search_relevant_chunks(self, query, top_k=15):
        """Search for relevant chunks using semantic similarity with keyword boosting"""
        query_embedding = self.embedding_model.encode([query])

        # Calculate semantic similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        similarities = similarities / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))

        # Add keyword boosting for form-specific queries
        form_pattern = r'([A-Z]{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})'
        form_match = re.search(form_pattern, query.replace('-', '_').upper())

        if form_match:
            form_code = form_match.group(1)
            logger.info(f"Applying keyword boost for form: {form_code}")

            # Boost chunks that contain the exact form code
            boost_applied = 0
            for i, metadata in enumerate(self.chunks_metadata):
                doc_name = metadata.get('document_name', '').upper()

                # Check various form code formats in document name
                form_variations = [
                    form_code,
                    form_code.replace('_', '-'),
                    form_code.replace('-', '_'),
                    form_code.replace('_', ' ').replace('-', ' ')
                ]

                for form_var in form_variations:
                    if form_var in doc_name:
                        similarities[i] += 0.5  # Significant boost for exact matches
                        boost_applied += 1
                        logger.info(
                            f"Keyword boosted chunk {i} from {doc_name[:50]}... (new score: {similarities[i]:.4f})")
                        break

            logger.info(f"Applied keyword boost to {boost_applied} chunks")

        # Get top indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        relevant_chunks = []
        for i, idx in enumerate(top_indices):
            if idx < len(self.chunks_metadata):  # Valid index
                # Get chunk content
                chunk_content = self._get_chunk_content(idx)
                metadata = self.chunks_metadata[idx]

                chunk_data = {
                    'chunk': chunk_content,
                    'metadata': metadata,
                    'score': float(similarities[idx])
                }
                relevant_chunks.append(chunk_data)

        return relevant_chunks

    def _get_chunk_content(self, chunk_idx):
        """Get chunk content by index"""
        # Try to load chunks from chunks.pkl file
        try:
            chunks_files = [
                os.path.join(self.embeddings_dir, "chunks.pkl"),
                os.path.join(self.embeddings_dir, "chunks")
            ]

            for chunks_file in chunks_files:
                if os.path.exists(chunks_file):
                    with open(chunks_file, 'rb') as f:
                        chunks = pickle.load(f)
                    if chunk_idx < len(chunks):
                        return chunks[chunk_idx]
                    break

            # Fallback: check if content is in metadata
            if chunk_idx < len(self.chunks_metadata):
                metadata = self.chunks_metadata[chunk_idx]
                if 'content' in metadata:
                    return metadata['content']

            # Last resort fallback
            return f"Content for chunk {chunk_idx} - unable to retrieve chunk text"

        except Exception as e:
            logger.error(f"Error retrieving chunk content: {e}")
            return f"Error retrieving chunk {chunk_idx}: {str(e)}"

    def _boost_form_specific_chunks(self, chunks, form_code, boost_factor=0.5):
        """Boost relevance scores for chunks from specific forms"""
        boosted_chunks = []
        form_specific_count = 0

        logger.info(f"Looking for form_code: {form_code}")

        for chunk_data in chunks:
            chunk = chunk_data.copy()
            source_file = chunk['metadata'].get('document_name', '').upper()

            logger.info(f"Checking source: {source_file}")

            # Enhanced matching logic
            form_variations = [
                form_code.upper(),
                form_code.replace('_', '-').upper(),
                form_code.replace('-', '_').upper(),
                form_code.replace('_', '').upper(),
                form_code.replace('-', '').upper()
            ]

            matches = False
            for form_var in form_variations:
                if form_var in source_file:
                    matches = True
                    logger.info(f"  ✅ MATCH found with variation: {form_var}")
                    break

            if matches:
                chunk['score'] += boost_factor  # Higher score = higher relevance
                form_specific_count += 1
                logger.info(f"  Boosted chunk from {source_file} (new score: {chunk['score']})")

            boosted_chunks.append(chunk)

        # Sort by boosted scores (descending)
        boosted_chunks.sort(key=lambda x: x['score'], reverse=True)

        return boosted_chunks, form_specific_count

    def _generate_claude_response(self, query, context_chunks, query_info):
        """Generate response using Claude with the retrieved context"""

        # Prepare context from chunks (reduced for speed)
        context_parts = []
        sources = set()

        for chunk_data in context_chunks[:5]:  # Reduced from 8 to 5 for faster processing
            chunk_text = chunk_data['chunk']
            source = chunk_data['metadata'].get('document_name', 'Unknown')
            sources.add(source)
            context_parts.append(f"Source: {source}\nContent: {chunk_text}\n")

        context = "\n---\n".join(context_parts)

        # Debug: Log what context is being sent to Claude
        logger.info(f"Context being sent to Claude ({len(context)} chars):")
        logger.info(f"Context preview: {context[:500]}...")
        logger.info(f"Number of context parts: {len(context_parts)}")
        for i, part in enumerate(context_parts):
            source_line = part.split('\n')[0] if '\n' in part else 'No source'
            content_preview = part[:100].replace('\n', ' ')
            logger.info(f"  Part {i + 1}: {source_line} - {content_preview}...")

        # Create specialized prompt based on query type
        if query_info['type'] == 'form_specific':
            system_prompt = """You are an expert insurance advisor specializing in commercial insurance forms and coverage analysis. 
            Provide detailed, accurate information based on the provided insurance documents. Focus on practical implications and coverage details.
            Use clear, professional language that is accessible to both insurance professionals and policyholders."""

            user_prompt = f"""Based on the following insurance documentation, please provide a comprehensive answer to this question: {query}

INSURANCE DOCUMENTATION:
{context}

Please provide:
1. A clear, detailed explanation of the form/coverage
2. Key coverage highlights and benefits
3. Important limitations, exclusions, or conditions
4. Practical implications for policyholders
5. Any specific requirements or procedures

Keep your response informative, well-structured, and accessible."""

        else:
            system_prompt = """You are a knowledgeable insurance expert who helps people understand their insurance coverage and policies. 
            Provide clear, accurate, and helpful information based on the provided documentation. Use professional but accessible language."""

            user_prompt = f"""Question: {query}

Based on this insurance documentation:
{context}

Please provide a clear, comprehensive, and helpful answer. If the documentation doesn't contain enough information to fully answer the question, please indicate what information is available and what might need additional clarification."""

        try:
            # Hard-coded API key as backup (replace with your actual key)
            backup_api_key = "your-working-api-key-here"  # Replace this!

            # Ensure API key is still set before making the call
            api_key = os.environ.get('ANTHROPIC_API_KEY') or backup_api_key
            if not api_key or not api_key.startswith('sk-ant-'):
                logger.error("API key lost during processing!")
                return {
                    'answer': "API key configuration error - please check system settings.",
                    'sources': list(sources),
                    'context_length': len(context),
                    'chunks_used': len(context_chunks[:5])
                }

            logger.info(f"Making API call with key: {api_key[:20]}...")

            # Create a fresh client for this request to avoid any state issues
            fresh_client = Anthropic(api_key=api_key)

            response = fresh_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=800,  # Reduced from 1500 for faster responses
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            return {
                'answer': response.content[0].text,
                'sources': list(sources),
                'context_length': len(context),
                'chunks_used': len(context_chunks[:5])  # Updated to match actual usage
            }

        except Exception as e:
            logger.error(f"Error generating Claude response: {str(e)}")
            return {
                'answer': f"I found relevant information but encountered an error generating the response: {str(e)}",
                'sources': list(sources),
                'context_length': len(context),
                'chunks_used': len(context_chunks[:5])  # Updated to match actual usage
            }

    def answer_question(self, query):
        """Main method to answer questions using the RAG system"""
        start_time = datetime.now()

        try:
            logger.info(f"Processing query: {query[:100]}...")

            # Analyze query
            query_info = self._detect_query_type(query)
            logger.info(f"Query type detected: {query_info['type']}")

            # Search for relevant chunks (reduced for speed)
            relevant_chunks = self._search_relevant_chunks(query, top_k=20)  # Reduced from 30

            # Debug: Log search results
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            for i, chunk in enumerate(relevant_chunks[:3]):  # Log top 3
                source = chunk['metadata'].get('document_name', 'Unknown')
                score = chunk['score']
                content_preview = chunk['chunk'][:100] if chunk['chunk'] else 'No content'
                logger.info(f"  Chunk {i + 1}: {source} (score: {score:.3f}) - {content_preview}...")

            # Apply form-specific boosting if applicable
            if query_info['type'] == 'form_specific':
                form_code = query_info['form_code']
                logger.info(f"Boosting for form: {form_code}")
                relevant_chunks, boosted_count = self._boost_form_specific_chunks(
                    relevant_chunks, form_code
                )
                logger.info(f"Boosted {boosted_count} chunks")

                # Debug: Log boosted results
                logger.info("Top chunks after boosting:")
                for i, chunk in enumerate(relevant_chunks[:3]):
                    source = chunk['metadata'].get('document_name', 'Unknown')
                    score = chunk['score']
                    logger.info(f"  Boosted {i + 1}: {source} (score: {score:.3f})")

            # Generate response with Claude
            response_data = self._generate_claude_response(query, relevant_chunks, query_info)

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()

            # Determine confidence based on relevance scores
            top_scores = [chunk['score'] for chunk in relevant_chunks[:5]]
            avg_score = np.mean(top_scores) if top_scores else 0.0
            confidence = max(0, min(100, int(avg_score * 100)))  # Convert to percentage

            logger.info(f"Response generated in {response_time:.2f}s with {confidence}% confidence")

            return {
                'answer': response_data['answer'],
                'sources': response_data['sources'],
                'confidence': confidence,
                'response_time': response_time,
                'query_type': query_info['type'],
                'chunks_used': response_data['chunks_used'],
                'success': True
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'confidence': 0,
                'response_time': (datetime.now() - start_time).total_seconds(),
                'query_type': 'error',
                'chunks_used': 0,
                'success': False,
                'error': str(e)
            }


# Initialize the RAG system with additional error handling
print("🚀 Initializing RAG system...")
try:
    logger.info("Initializing RAG system...")
    rag_system = ClaudeInsuranceRAG()
    logger.info("RAG system initialized successfully!")
    print("✅ RAG system loaded successfully!")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    print(f"❌ RAG system failed: {e}")
    rag_system = None

# Test API key again after RAG initialization
print("🧪 Testing API key after RAG initialization...")
try:
    test_client = Anthropic()
    test_response = test_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=5,
        messages=[{"role": "user", "content": "Test"}]
    )
    print("✅ API key still works after RAG init!")
except Exception as e:
    print(f"❌ API key failed after RAG init: {e}")


@app.route('/test-specific-form/<form_code>')
def test_specific_form(form_code):
    """Test with manually selected chunks for a specific form"""
    if not rag_system:
        return jsonify({'error': 'RAG system not available'})

    try:
        # Manually find chunks from the specific form
        form_chunks = []
        for i, metadata in enumerate(rag_system.chunks_metadata):
            doc_name = metadata.get('document_name', '')
            if form_code.upper() in doc_name.upper():
                chunk_content = rag_system._get_chunk_content(i)
                form_chunks.append({
                    'chunk': chunk_content,
                    'metadata': metadata,
                    'score': 1.0  # High score for manual selection
                })

        if not form_chunks:
            return jsonify({
                'error': f'No chunks found for form {form_code}',
                'available_forms': [doc for doc in set(m['document_name'] for m in rag_system.chunks_metadata) if
                                    'BM_99' in doc]
            })

        # Generate response using only these chunks
        query = f"What is {form_code} all about?"
        query_info = {'type': 'form_specific', 'form_code': form_code}

        response_data = rag_system._generate_claude_response(query, form_chunks, query_info)

        return jsonify({
            'query': query,
            'chunks_found': len(form_chunks),
            'sources': [chunk['metadata']['document_name'] for chunk in form_chunks],
            'response': response_data['answer'],
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/debug-form/<form_code>')
def debug_form_search(form_code):
    """Debug endpoint to test form-specific search"""
    if not rag_system:
        return jsonify({'error': 'RAG system not available'})

    try:
        # Test the search for this specific form
        query = f"What is {form_code} all about?"

        # Get query info
        query_info = rag_system._detect_query_type(query)

        # Search for chunks
        chunks = rag_system._search_relevant_chunks(query, top_k=10)

        # Apply boosting if form-specific
        if query_info['type'] == 'form_specific':
            boosted_chunks, boost_count = rag_system._boost_form_specific_chunks(chunks, form_code)
        else:
            boosted_chunks = chunks
            boost_count = 0

        # Get the actual content that would be sent to Claude
        context_parts = []
        for chunk_data in boosted_chunks[:5]:
            chunk_text = chunk_data['chunk']
            source = chunk_data['metadata'].get('document_name', 'Unknown')
            context_parts.append(f"Source: {source}\nContent: {chunk_text}\n")

        return jsonify({
            'query': query,
            'query_type': query_info['type'],
            'form_detected': query_info.get('form_code'),
            'total_chunks_found': len(chunks),
            'boosted_chunks': boost_count,
            'context_parts': len(context_parts),
            'top_sources': [chunk['metadata'].get('document_name') for chunk in boosted_chunks[:5]],
            'top_scores': [chunk['score'] for chunk in boosted_chunks[:5]],
            'sample_content': context_parts[0][:500] if context_parts else 'No content'
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/simple-test')
def simple_test():
    """Simple test endpoint without RAG"""
    return jsonify({
        'message': 'Flask is working!',
        'rag_loaded': rag_system is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/simple-chat', methods=['POST'])
def simple_chat():
    """Simple chat without RAG for testing"""
    try:
        data = request.get_json()
        message = data.get('message', 'No message')

        return jsonify({
            'response': f'You said: {message}',
            'success': True,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'response': f'Error: {str(e)}',
            'success': False
        })


@app.route('/api-test', methods=['POST'])
def api_test():
    """Test just the API without RAG search"""
    try:
        data = request.get_json()
        message = data.get('message', 'Hello')

        # Direct API call without RAG
        client = Anthropic()
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": f"Briefly answer: {message}"}]
        )

        return jsonify({
            'response': response.content[0].text,
            'success': True,
            'message_received': message
        })

    except Exception as e:
        return jsonify({
            'response': f'API Error: {str(e)}',
            'success': False
        })


@app.route('/')
def home():
    """Serve the main chat interface"""
    return render_template('insurance_chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        user_message = request.json.get('message', '').strip()

        if not user_message:
            return jsonify({
                'response': 'Please ask me a question about insurance.',
                'sources': [],
                'confidence': 0,
                'success': False
            })

        if not rag_system:
            return jsonify({
                'response': 'Sorry, the insurance knowledge system is not available right now. Please try again later.',
                'sources': [],
                'confidence': 0,
                'success': False
            })

        # Get response from RAG system
        result = rag_system.answer_question(user_message)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': f'Sorry, I encountered an error: {str(e)}',
            'sources': [],
            'confidence': 0,
            'success': False,
            'error': str(e)
        })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_system_loaded': rag_system is not None,
        'chunks_count': len(rag_system.chunks_metadata) if rag_system else 0,
        'config': rag_system.config if rag_system else {}
    })


@app.route('/stats')
def get_stats():
    """Get system statistics"""
    if not rag_system:
        return jsonify({'error': 'RAG system not available'})

    return jsonify({
        'total_chunks': len(rag_system.chunks_metadata),
        'total_documents': len(set([meta['document_name'] for meta in rag_system.chunks_metadata])),
        'embedding_model': rag_system.config.get('model_name', 'unknown'),
        'chunking_method': rag_system.config.get('chunking_method', 'unknown'),
        'sentences_per_chunk': rag_system.config.get('sentences_per_chunk', 'unknown'),
        'overlap_sentences': rag_system.config.get('overlap_sentences', 'unknown')
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)