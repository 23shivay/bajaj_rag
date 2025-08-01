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
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logging.getLogger("pinecone").setLevel(logging.WARNING)

# --- 0. Load Environment Variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

if not groq_api_key or not pinecone_api_key:
    raise ValueError("Please ensure GROQ_API_KEY and PINECONE_API_KEY are set in your .env file")

logging.info("Environment variables loaded.")

pc = None
index = None
vectorstore = None
embedding_model = None
sparse_encoder = None
index_name = "insurance-langchain-enhanced"

def initialize_models():
    global embedding_model, sparse_encoder, pc
    
    if embedding_model is None:
        logging.info("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    if sparse_encoder is None:
        logging.info("Initializing sparse encoder...")
        sparse_encoder = BM25Encoder()
        sparse_encoder.fit(["dummy"])
    
    if pc is None:
        pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)

def create_memory_efficient_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
        keep_separator=False
    )

def download_and_process_document_optimized(document_url: str) -> List[Document]:
    if not document_url:
        return []

    logging.info(f"Processing: {document_url}")
    temp_file_path = None
    
    try:
        response = requests.get(document_url, stream=True)
        response.raise_for_status()

        file_hash = hashlib.md5(document_url.encode('utf-8')).hexdigest()[:8]
        parsed_url = urlparse(document_url)
        filename_base = os.path.basename(parsed_url.path) or "doc"
        temp_file_path = f"temp_{file_hash}.pdf"

        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
        
        del response
        gc.collect()

        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()
        
        splitter = create_memory_efficient_splitter()
        chunks = splitter.split_documents(docs)
        
        del docs
        gc.collect()
        
        logging.info(f"Processed into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logging.error(f"Error processing document: {e}")
        return []
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

def setup_pinecone_optimized(document_url_hash: str, chunks_to_upsert: List[Document]):
    global index
    
    initialize_models()
    
    if index_name not in pc.list_indexes().names():
        logging.info(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(2)

    index = pc.Index(index_name)

    logging.info(f"Upserting {len(chunks_to_upsert)} chunks...")
    
    BATCH_SIZE = 50
    
    for i in range(0, len(chunks_to_upsert), BATCH_SIZE):
        batch_chunks = chunks_to_upsert[i:i + BATCH_SIZE]
        vectors_to_upsert = []
        
        for j, chunk in enumerate(batch_chunks):
            try:
                dense_vector = embedding_model.embed_query(chunk.page_content)
                sparse_vector_data = sparse_encoder.encode_queries([chunk.page_content])[0]

                metadata = {
                    'text': chunk.page_content[:1000],
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
                logging.error(f"Error processing chunk {i+j}: {e}")
                continue

        if vectors_to_upsert:
            try:
                index.upsert(vectors=vectors_to_upsert)
                logging.info(f"Upserted batch {i//BATCH_SIZE + 1}")
            except Exception as e:
                logging.error(f"Error upserting batch: {e}")
        
        del vectors_to_upsert
        gc.collect()

def pre_index_documents(document_url: str):
    """
    New function to handle the one-time, pre-hackathon indexing.
    """
    if not document_url:
        logging.info("No document URL provided for pre-indexing. Skipping.")
        return

    logging.info("--- Starting one-time pre-indexing of documents ---")
    document_hash = hashlib.md5(document_url.encode('utf-8')).hexdigest()[:8]
    document_chunks = download_and_process_document_optimized(document_url)
    
    if not document_chunks:
        logging.error("Pre-indexing failed. Could not process document.")
        return
    
    setup_pinecone_optimized(document_hash, document_chunks)
    
    del document_chunks
    gc.collect()
    logging.info("--- Pre-indexing complete ---")


# --- LLM and Prompt Definitions ---
class InternalSearchQueryItem(BaseModel):
    type: Literal["keyword", "semantic"] = Field(description="Search type")
    query: str = Field(description="Search query")
    reason: str = Field(description="Search reason")

class InternalSearchPlan(BaseModel):
    searches: List[InternalSearchQueryItem] = Field(description="Search queries")

class AnswersResponse(BaseModel):
    answers: List[str] = Field(description="Generated answers")

def generate_simple_search_plan(question: str) -> List[InternalSearchQueryItem]:
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

def perform_optimized_search(search_query_item, k: int = 4, document_filter: Optional[dict] = None) -> List[Document]:
    try:
        retrieved_docs = vectorstore.invoke(
            search_query_item.query,
            config={"configurable": {"search_kwargs": {"k": k, "filter": document_filter}}}
        )
        return retrieved_docs
    except Exception as e:
        logging.error(f"Search error: {e}")
        return []

def simple_parse_answer(raw_response: str) -> str:
    try:
        if '{' in raw_response and 'answers' in raw_response:
            json_match = re.search(r'\{[^{}]*"answers"[^{}]*\[[^\]]*\][^{}]*\}', raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'answers' in data and data['answers']:
                    return str(data['answers'][0]).strip()
        
        cleaned = raw_response.strip().strip('"{}[]')
        if len(cleaned) > 5 and len(cleaned) < 500:
            return cleaned
            
    except Exception as e:
        logging.error(f"Parse error: {e}")
    
    return "Information not found"


answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer based on context. Output as JSON: {{"answers": ["your answer"]}}
If not found, return: {{"answers": ["Information not found"]}}

Context: {context}"""),
    ("human", "Question: {question_text}")
])


def run_optimized_pipeline(questions: List[str]):
    """Memory-optimized pipeline for live Q&A"""
    logging.info(f"Processing {len(questions)} questions...")
    
    try:
        initialize_models()
        model = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
        answer_chain = answer_prompt | model
        
        # Initialize retriever after pre-indexing is complete
        global vectorstore
        if vectorstore is None:
            initialize_models()
            index = pc.Index(index_name)
            vectorstore = PineconeHybridSearchRetriever(
                embeddings=embedding_model,
                sparse_encoder=sparse_encoder,
                index=index,
                text_key="text",
            )
            logging.info("Retriever initialized for live Q&A.")

        final_answers = []
        
        for question in questions:
            logging.info("-" * 50)
            logging.info(f"Processing question: {question}")
            
            try:
                searches = generate_simple_search_plan(question)
                # Verbose log removed: logging.info(f"-> Generated search queries: {searches}")
                
                all_docs = []
                unique_doc_contents = set()
                
                for search_item in searches:
                    docs = perform_optimized_search(search_item, k=4)
                    for doc in docs:
                        if doc.page_content not in unique_doc_contents:
                            all_docs.append(doc)
                            unique_doc_contents.add(doc.page_content)
                
                if not all_docs:
                    logging.warning("-> No documents retrieved from Pinecone.")
                    final_answers.append("Information not found")
                    continue
                
                logging.info(f"-> Retrieved {len(all_docs)} unique documents.")
                # Verbose loop removed: printing contents of each doc
                
                context_parts = []
                total_length = 0
                for doc in all_docs:
                    chunk = doc.page_content[:500]
                    if total_length + len(chunk) > 2500:
                        logging.warning("-> Context limit reached. Skipping remaining documents.")
                        break
                    context_parts.append(chunk)
                    total_length += len(chunk)

                context_str = "\n\n".join(context_parts)
                # Verbose log removed: printing the full context string. Only a summary is kept.
                logging.info(f"-> Final context sent to LLM (total length: {len(context_str)})")
                
                raw_response = answer_chain.invoke({
                    "context": context_str,
                    "question_text": question
                })
                
                # Verbose log removed: printing the full raw LLM response.
                
                answer = simple_parse_answer(raw_response.content)
                final_answers.append(answer)
                logging.info(f"-> Final Answer: {answer}")
                
                del all_docs, context_str, raw_response, unique_doc_contents
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing question: {e}")
                final_answers.append("Processing error")
                traceback.print_exc()
        
        return {"answers": final_answers}
        
    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        traceback.print_exc()
        return {"answers": ["Pipeline error"] * len(questions)}