import os
from dotenv import load_dotenv
import time
import logging
import json
import requests
from urllib.parse import urlparse
import hashlib
import re
import gc
import tempfile
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# Minimal logging to save memory
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

if not groq_api_key or not pinecone_api_key:
    raise ValueError("Missing API keys in .env file")

print("Environment loaded.")

# Global variables - ultra-lazy initialization
pc = None
index = None
vectorstore = None
embedding_model = None
sparse_encoder = None
index_name = "ultra-optimized-rag"

def initialize_ultra_light_models():
    """Ultra-lightweight model initialization"""
    global embedding_model, sparse_encoder, pc
    
    if embedding_model is None:
        print("Loading ultra-light embedding model...")
        # Use the smallest possible model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Even smaller: 61MB vs 90MB
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': False
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 1,  # Process one at a time to save memory
                'show_progress_bar': False
            }
        )
    
    if sparse_encoder is None:
        print("Loading minimal sparse encoder...")
        sparse_encoder = BM25Encoder()
        # Fit with minimal data
        sparse_encoder.fit(["sample"])
    
    if pc is None:
        pc = Pinecone(api_key=pinecone_api_key)

def create_ultra_light_splitter():
    """Ultra-small chunks to minimize memory usage"""
    return RecursiveCharacterTextSplitter(
        chunk_size=200,  # Very small chunks
        chunk_overlap=20,  # Minimal overlap
        separators=["\n\n", "\n", ". "],
        keep_separator=False
    )

def download_and_process_ultra_light(document_url: str) -> List[Document]:
    """Ultra-lightweight document processing"""
    if not document_url:
        return []

    print(f"Processing document...")
    
    try:
        # Use temporary file with context manager
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Stream download with very small chunks
            response = requests.get(document_url, stream=True, timeout=30)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=1024):  # Very small chunks
                if chunk:
                    temp_file.write(chunk)
        
        # Clear response immediately
        del response
        gc.collect()

        # Load and process document
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        
        # Split into very small chunks
        splitter = create_ultra_light_splitter()
        chunks = splitter.split_documents(docs)
        
        # Limit number of chunks to save memory
        if len(chunks) > 100:  # Limit to 100 chunks max
            chunks = chunks[:100]
        
        # Clear docs immediately
        del docs, loader
        gc.collect()
        
        print(f"Created {len(chunks)} ultra-light chunks.")
        return chunks

    except Exception as e:
        print(f"Document processing error: {e}")
        return []
    finally:
        # Always cleanup temp file
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass

def setup_ultra_light_pinecone(document_hash: str, chunks: List[Document]):
    """Ultra-lightweight Pinecone setup"""
    global index, vectorstore
    
    initialize_ultra_light_models()
    
    # Create or get index
    if index_name not in pc.list_indexes().names():
        print(f"Creating ultra-light index...")
        pc.create_index(
            name=index_name,
            dimension=384,  # paraphrase-MiniLM-L3-v2 dimension
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
        )
        # Wait for readiness
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pc.Index(index_name)

    print(f"Upserting {len(chunks)} chunks in ultra-small batches...")
    
    # Ultra-small batch size to minimize memory usage
    ULTRA_BATCH_SIZE = 10
    
    for i in range(0, len(chunks), ULTRA_BATCH_SIZE):
        batch_chunks = chunks[i:i + ULTRA_BATCH_SIZE]
        vectors = []
        
        for j, chunk in enumerate(batch_chunks):
            try:
                # Process one embedding at a time
                dense_vec = embedding_model.embed_query(chunk.page_content)
                sparse_vec = sparse_encoder.encode_queries([chunk.page_content])[0]

                # Minimal metadata
                metadata = {
                    'text': chunk.page_content[:500],  # Truncate text
                    'doc_hash': document_hash,
                    'page': int(chunk.metadata.get('page', 0))
                }

                vectors.append({
                    "id": f"doc_{document_hash}_{i+j}",
                    "values": dense_vec,
                    "sparse_values": sparse_vec,
                    "metadata": metadata
                })

                # Clear variables immediately
                del dense_vec, sparse_vec
                
            except Exception as e:
                print(f"Error processing chunk {i+j}: {e}")
                continue

        # Upsert ultra-small batch
        if vectors:
            try:
                index.upsert(vectors=vectors)
                print(f"Upserted ultra-batch {i//ULTRA_BATCH_SIZE + 1}")
            except Exception as e:
                print(f"Upsert error: {e}")
        
        # Aggressive cleanup
        del vectors, batch_chunks
        gc.collect()

    # Initialize retriever
    vectorstore = PineconeHybridSearchRetriever(
        embeddings=embedding_model,
        sparse_encoder=sparse_encoder,
        index=index,
        text_key="text"
    )
    print("Ultra-light retriever ready.")

# Simplified models
class UltraSearchItem(BaseModel):
    query: str = Field(description="Search query")

def generate_ultra_simple_searches(question: str) -> List[str]:
    """Ultra-simple search generation"""
    # Extract key terms
    words = re.findall(r'\b\w{3,}\b', question.lower())
    key_words = [w for w in words if w not in {'what', 'does', 'this', 'policy', 'cover', 'the', 'are', 'any'}]
    
    searches = []
    if key_words:
        searches.append(' '.join(key_words[:3]))  # Keyword search
    searches.append(question)  # Semantic search
    
    return searches

def perform_ultra_light_search(query: str, k: int = 2, doc_filter: Optional[dict] = None) -> List[Document]:
    """Ultra-lightweight search with minimal results"""
    try:
        docs = vectorstore.invoke(
            query,
            config={"configurable": {"search_kwargs": {"k": k, "filter": doc_filter}}}
        )
        return docs
    except Exception as e:
        print(f"Search error: {e}")
        return []

def ultra_simple_parse(response: str) -> str:
    """Ultra-simple response parsing"""
    try:
        # Try to extract JSON answer
        if '"answers"' in response and '[' in response:
            start = response.find('[')
            end = response.find(']', start) + 1
            if start != -1 and end != 0:
                answers_list = json.loads(response[start:end])
                if answers_list:
                    return str(answers_list[0]).strip()
        
        # Fallback: clean the response
        cleaned = response.strip().strip('"{}[]')
        if 10 < len(cleaned) < 300:
            return cleaned
            
    except:
        pass
    
    return "Information not available"

# Ultra-simple prompts
ultra_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer briefly based on context. Format: {\"answers\": [\"your answer\"]}\n\nContext: {context}"),
    ("human", "Question: {question}")
])

def run_ultra_optimized_pipeline(questions: List[str], documents_url: Optional[str] = None):
    """Ultra-memory-optimized pipeline for 512MB constraint"""
    print(f"Ultra-processing {len(questions)} questions...")
    
    if not documents_url:
        return {"answers": ["No document provided"] * len(questions)}
    
    try:
        # Process document with ultra-light approach
        doc_hash = hashlib.md5(documents_url.encode()).hexdigest()[:6]  # Shorter hash
        chunks = download_and_process_ultra_light(documents_url)
        
        if not chunks:
            return {"answers": ["Document processing failed"] * len(questions)}
        
        # Setup ultra-light Pinecone
        setup_ultra_light_pinecone(doc_hash, chunks)
        
        # Clear chunks immediately
        del chunks
        gc.collect()
        
        doc_filter = {"doc_hash": doc_hash}
        
        # Initialize ultra-light model
        initialize_ultra_light_models()
        model = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama3-8b-8192",
            temperature=0,
            max_tokens=150  # Limit response length
        )
        answer_chain = ultra_answer_prompt | model
        
        final_answers = []
        
        # Process questions one by one to minimize memory
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}...")
            
            try:
                # Generate ultra-simple searches
                search_queries = generate_ultra_simple_searches(question)
                
                # Perform minimal searches
                all_docs = []
                for query in search_queries[:2]:  # Limit to 2 searches
                    docs = perform_ultra_light_search(query, k=2, doc_filter=doc_filter)
                    all_docs.extend(docs)
                
                if not all_docs:
                    final_answers.append("Information not found")
                    continue
                
                # Create minimal context
                context_parts = []
                total_len = 0
                for doc in all_docs[:4]:  # Limit to 4 docs
                    if total_len > 800:  # Very small context limit
                        break
                    text = doc.page_content[:200]  # Truncate each doc
                    context_parts.append(text)
                    total_len += len(text)
                
                context = "\n".join(context_parts)
                
                # Generate answer
                response = answer_chain.invoke({
                    "context": context,
                    "question": question
                })
                
                answer = ultra_simple_parse(response.content)
                final_answers.append(answer)
                
                # Aggressive cleanup after each question
                del all_docs, context_parts, context, response
                gc.collect()
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                final_answers.append("Processing error")
        
        return {"answers": final_answers}
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        return {"answers": ["Pipeline error"] * len(questions)}

# Main execution
if __name__ == "__main__":
    print("Starting ultra-optimized RAG for 512MB memory...")
    
    sample_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?"
        ]
    }
    
    try:
        result = run_ultra_optimized_pipeline(
            questions=sample_request["questions"],
            documents_url=sample_request["documents"]
        )
        print("\n--- Ultra-Optimized Results ---")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Execution error: {e}")
        import traceback
        traceback.print_exc()