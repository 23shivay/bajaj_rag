import os
from dotenv import load_dotenv
import time
import logging
import traceback
import json
import requests
from urllib.parse import urlparse
import hashlib
import re
import gc
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# Configure logging to reduce memory overhead
logging.basicConfig(level=logging.WARNING)

# --- 0. Load Environment Variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

if not groq_api_key or not pinecone_api_key:
    raise ValueError("Please ensure GROQ_API_KEY and PINECONE_API_KEY are set in your .env file")

print("Environment variables loaded.")

# Global variables - initialized lazily
pc = None
index = None
vectorstore = None
embedding_model = None
sparse_encoder = None
index_name = "insurance-langchain-enhanced"

# --- MEMORY-OPTIMIZED INITIALIZATION ---
def initialize_models():
    """Lazy initialization of models to save memory"""
    global embedding_model, sparse_encoder, pc
    
    if embedding_model is None:
        print("Initializing embedding model...")
        # Use a smaller model to reduce memory usage
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU memory issues
            encode_kwargs={'normalize_embeddings': True}
        )
    
    if sparse_encoder is None:
        print("Initializing sparse encoder...")
        sparse_encoder = BM25Encoder()
        # Minimal fitting to reduce memory
        sparse_encoder.fit(["dummy"])
    
    if pc is None:
        pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)

# --- MEMORY-OPTIMIZED TEXT SPLITTER ---
def create_memory_efficient_splitter():
    """Smaller chunks to reduce memory usage"""
    return RecursiveCharacterTextSplitter(
        chunk_size=400,  # Reduced from 800
        chunk_overlap=50,  # Reduced from 100
        separators=["\n\n", "\n", ". ", " "],
        keep_separator=False  # Save memory
    )

# --- OPTIMIZED DOCUMENT PROCESSING ---
def download_and_process_document_optimized(document_url: str) -> List[Document]:
    """Memory-optimized document processing with cleanup"""
    if not document_url:
        return []

    print(f"Processing: {document_url}")
    temp_file_path = None
    
    try:
        # Stream download with smaller chunks
        response = requests.get(document_url, stream=True)
        response.raise_for_status()

        file_hash = hashlib.md5(document_url.encode('utf-8')).hexdigest()[:8]  # Shorter hash
        parsed_url = urlparse(document_url)
        filename_base = os.path.basename(parsed_url.path) or "doc"
        temp_file_path = f"temp_{file_hash}.pdf"

        # Write with smaller buffer
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=4096):  # Smaller chunks
                if chunk:
                    f.write(chunk)
        
        # Clear response from memory
        del response
        gc.collect()

        # Process document
        content_type = 'pdf'  # Assume PDF for simplicity
        docs = []
        
        if content_type == 'pdf':
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
        
        # Use memory-efficient splitter
        splitter = create_memory_efficient_splitter()
        chunks = splitter.split_documents(docs)
        
        # Clear docs from memory
        del docs
        gc.collect()
        
        print(f"Processed into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        print(f"Error processing document: {e}")
        return []
    finally:
        # Always cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

# --- OPTIMIZED PINECONE SETUP ---
def setup_pinecone_optimized(document_url_hash: str, chunks_to_upsert: List[Document]):
    """Memory-optimized Pinecone setup with batching"""
    global index, vectorstore
    
    initialize_models()
    
    # Create index if needed
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(2)

    index = pc.Index(index_name)

    print(f"Upserting {len(chunks_to_upsert)} chunks...")
    
    # Process in smaller batches to reduce memory usage
    BATCH_SIZE = 50  # Reduced from 100
    
    for i in range(0, len(chunks_to_upsert), BATCH_SIZE):
        batch_chunks = chunks_to_upsert[i:i + BATCH_SIZE]
        vectors_to_upsert = []
        
        for j, chunk in enumerate(batch_chunks):
            try:
                # Generate embeddings
                dense_vector = embedding_model.embed_query(chunk.page_content)
                sparse_vector_data = sparse_encoder.encode_queries([chunk.page_content])[0]

                # Minimal metadata to save memory
                metadata = {
                    'text': chunk.page_content[:1000],  # Truncate to save space
                    'document_url_hash': document_url_hash,
                    'page': int(chunk.metadata.get('page', 0))
                }

                vector_id = f"doc_{document_url_hash}_chunk_{i+j}"

                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": dense_vector,
                    "sparse_values": sparse_vector_data,
                    "metadata": metadata
                })

            except Exception as e:
                print(f"Error processing chunk {i+j}: {e}")
                continue

        # Upsert batch
        if vectors_to_upsert:
            try:
                index.upsert(vectors=vectors_to_upsert)
                print(f"Upserted batch {i//BATCH_SIZE + 1}")
            except Exception as e:
                print(f"Error upserting batch: {e}")
        
        # Clear batch from memory
        del vectors_to_upsert
        gc.collect()

    # Initialize retriever
    vectorstore = PineconeHybridSearchRetriever(
        embeddings=embedding_model,
        sparse_encoder=sparse_encoder,
        index=index,
        text_key="text",
    )
    print("Retriever initialized.")

# --- SIMPLIFIED MODELS ---
class InternalSearchQueryItem(BaseModel):
    type: Literal["keyword", "semantic"] = Field(description="Search type")
    query: str = Field(description="Search query")
    reason: str = Field(description="Search reason")

class InternalSearchPlan(BaseModel):
    searches: List[InternalSearchQueryItem] = Field(description="Search queries")

class AnswersResponse(BaseModel):
    answers: List[str] = Field(description="Generated answers")

# --- SIMPLIFIED SEARCH LOGIC ---
def generate_simple_search_plan(question: str) -> List[InternalSearchQueryItem]:
    """Generate a simple search plan to reduce processing overhead"""
    question_lower = question.lower()
    key_terms = re.findall(r'\b\w+\b', question_lower)
    key_terms = [word for word in key_terms if len(word) > 2 and word not in 
                {'what', 'is', 'the', 'are', 'there', 'any', 'does', 'this', 'policy', 'under'}]
    
    searches = [
        InternalSearchQueryItem(
            type="keyword",
            query=' '.join(key_terms[:3]),
            reason="Direct keyword search"
        ),
        InternalSearchQueryItem(
            type="semantic",
            query=question,
            reason="Semantic search"
        )
    ]
    
    return searches

# --- OPTIMIZED SEARCH FUNCTION ---
def perform_optimized_search(search_query_item, k: int = 4, document_filter: Optional[dict] = None) -> List[Document]:
    """Reduced k value to limit memory usage"""
    try:
        retrieved_docs = vectorstore.invoke(
            search_query_item.query,
            config={"configurable": {"search_kwargs": {"k": k, "filter": document_filter}}}
        )
        return retrieved_docs
    except Exception as e:
        print(f"Search error: {e}")
        return []

# --- OPTIMIZED PARSING ---
def simple_parse_answer(raw_response: str) -> str:
    """Simplified answer parsing"""
    try:
        # Try JSON first
        if '{' in raw_response and 'answers' in raw_response:
            json_match = re.search(r'\{[^{}]*"answers"[^{}]*\[[^\]]*\][^{}]*\}', raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'answers' in data and data['answers']:
                    return str(data['answers'][0]).strip()
        
        # Fallback to cleaning the response
        cleaned = raw_response.strip().strip('"{}[]')
        if len(cleaned) > 5 and len(cleaned) < 500:
            return cleaned
            
    except Exception as e:
        print(f"Parse error: {e}")
    
    return "Information not found"

# --- OPTIMIZED PROMPTS (SHORTER) ---
search_prompt = ChatPromptTemplate.from_messages([
    ("system", """Generate 2 search queries for this question. Output as JSON:
{"searches": [{"type": "keyword", "query": "...", "reason": "..."}, {"type": "semantic", "query": "...", "reason": "..."}]}"""),
    ("human", "Question: {input}")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer based on context. Output as JSON: {"answers": ["your answer"]}
If not found, return: {"answers": ["Information not found"]}

Context: {context}"""),
    ("human", "Question: {question_text}")
])

# --- MAIN OPTIMIZED PIPELINE ---
def run_optimized_pipeline(questions: List[str], documents_url: Optional[str] = None):
    """Memory-optimized pipeline"""
    print(f"Processing {len(questions)} questions...")
    
    if not documents_url:
        return {"answers": ["No document provided"] * len(questions)}
    
    try:
        # Process document
        document_hash = hashlib.md5(documents_url.encode('utf-8')).hexdigest()[:8]
        document_chunks = download_and_process_document_optimized(documents_url)
        
        if not document_chunks:
            return {"answers": ["Document processing failed"] * len(questions)}
        
        # Setup Pinecone
        setup_pinecone_optimized(document_hash, document_chunks)
        
        # Clear chunks from memory
        del document_chunks
        gc.collect()
        
        document_filter = {"document_url_hash": document_hash}
        
        # Initialize models for chain
        initialize_models()
        model = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")  # Smaller model
        search_chain = search_prompt | model
        answer_chain = answer_prompt | model
        
        final_answers = []
        
        for question in questions:
            print(f"Processing: {question[:50]}...")
            
            try:
                # Generate simple search plan
                searches = generate_simple_search_plan(question)
                
                # Perform searches
                all_docs = []
                for search_item in searches:
                    docs = perform_optimized_search(search_item, k=3, document_filter=document_filter)
                    all_docs.extend(docs)
                
                if not all_docs:
                    final_answers.append("Information not found")
                    continue
                
                # Prepare context (limit size)
                context_parts = []
                total_length = 0
                for doc in all_docs:
                    if total_length > 2000:  # Limit context size
                        break
                    context_parts.append(doc.page_content[:500])  # Truncate each doc
                    total_length += len(doc.page_content)
                
                context_str = "\n\n".join(context_parts)
                
                # Generate answer
                raw_response = answer_chain.invoke({
                    "context": context_str,
                    "question_text": question
                })
                
                answer = simple_parse_answer(raw_response.content)
                final_answers.append(answer)
                
                # Clear variables
                del all_docs, context_str, raw_response
                gc.collect()
                
            except Exception as e:
                print(f"Error processing question: {e}")
                final_answers.append("Processing error")
        
        return {"answers": final_answers}
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        return {"answers": ["Pipeline error"] * len(questions)}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Starting optimized RAG pipeline...")
    
    sample_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?",
            "What is the waiting period for cataract surgery?",
            "Are medical expenses for organ donors covered?"
        ]
    }
    
    try:
        result = run_optimized_pipeline(
            questions=sample_request["questions"],
            documents_url=sample_request["documents"]
        )
        print("\n--- Final Results ---")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Execution error: {e}")
        traceback.print_exc()