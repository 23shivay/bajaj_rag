import os
from dotenv import load_dotenv
import time
import logging
import traceback
import json
import requests # For downloading from URL
from urllib.parse import urlparse # For parsing URL for filename
import hashlib # For hashing URL content to check if doc is new
import re

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Changed import for HuggingFaceEmbeddings as per deprecation warning
from langchain_huggingface import HuggingFaceEmbeddings # Ensure you pip install langchain-huggingface
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# --- 0. Load Environment Variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1") # Default if not set

if not groq_api_key or not pinecone_api_key:
    raise ValueError("Please ensure GROQ_API_KEY and PINECONE_API_KEY are set in your .env file")

print("Imports and environment variables loaded.")

# Global Pinecone index and retriever (will be re-initialized if document URL changes)
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = "insurance-langchain-enhanced" # Changed index name for enhanced version
index = None # Will be set dynamically
vectorstore = None # Will be set dynamically

# --- 1. Initialize Embedders (these are static) ---
print("\n--- Initializing Embedding Models ---")
try:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sparse_encoder = BM25Encoder()
    # Sparse encoder will be fitted dynamically or with a general corpus
    # For now, fitting with a dummy corpus; ideally, it's fitted once on a large, representative corpus.
    sparse_encoder.fit(["This is a dummy sentence for BM25 initialization.", "Another dummy sentence."])
    print("Embedding models initialized.")
except Exception as e:
    print(f"Error initializing embedding models: {e}")
    exit()

# --- ENHANCED TEXT SPLITTER ---
def create_smart_splitter():
    """Enhanced text splitter for insurance documents with larger chunks and better overlap"""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,  # Increased from 500
        chunk_overlap=100,  # Increased from 50
        separators=["\n\n", "\n", ". ", ", ", " "],
        keep_separator=True
    )

# --- QUESTION TYPE DETECTION ---
def detect_question_type(question: str) -> str:
    """Detect the type of question to customize search strategy"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["sub-limit", "limit", "charges", "percentage", "maximum", "minimum", "%", "amount"]):
        return "numerical_limits"
    elif any(word in question_lower for word in ["cover", "coverage", "benefit", "included", "excluded", "expenses"]):
        return "coverage_conditions"
    elif any(word in question_lower for word in ["waiting period", "time", "duration", "period", "grace period"]):
        return "time_based"
    elif any(word in question_lower for word in ["define", "definition", "means", "what is"]):
        return "definition"
    else:
        return "general"

# --- KEY TERMS EXTRACTION ---
def extract_key_terms(question: str) -> List[str]:
    """Extract key terms from question for enhanced search"""
    # Remove common question words
    stop_words = {"what", "is", "the", "are", "there", "any", "does", "this", "policy", "under", "for", "how", "and", "or", "a", "an"}
    
    # Split and clean
    words = re.findall(r'\b\w+\b', question.lower())
    key_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    return key_terms

def extract_main_topic(question: str) -> str:
    """Extract the main topic from a question"""
    key_terms = extract_key_terms(question)
    # Return the most relevant terms joined
    return " ".join(key_terms[:3])  # Take first 3 key terms

# --- Utility for Dynamic Document Handling ---
def download_and_process_document(document_url: str) -> List[Document]:
    """
    Downloads a document from a URL, determines its type, and processes it into chunks.
    Currently supports PDF and conceptual DOCX.
    ENHANCED: Uses smart splitter with larger chunks
    """
    if not document_url:
        return []

    print(f"Attempting to download and process: {document_url}")
    try:
        response = requests.get(document_url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        parsed_url = urlparse(document_url)
        # Use hashlib to create a unique filename based on the URL's content hash or a simple hash of the URL itself
        # For this example, we'll use a simple hash of the URL + timestamp for filename
        file_hash = hashlib.md5(document_url.encode('utf-8')).hexdigest()
        filename_base = os.path.basename(parsed_url.path)
        if not filename_base: # If path is just '/', use a generic name
            filename_base = "downloaded_document"
        temp_file_path = f"temp_doc_{file_hash}_{os.path.splitext(filename_base)[1] or '.pdf'}" # Default to .pdf if no extension

        # Determine file type
        content_type = response.headers.get('Content-Type', '').lower()
        file_extension = os.path.splitext(temp_file_path)[1].lower()

        chunks = []
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded content to: {temp_file_path}")

        docs = []
        if 'pdf' in content_type or file_extension == '.pdf':
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
        elif 'word' in content_type or file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
            docs = loader.load()
        elif 'message/rfc822' in content_type or file_extension == '.eml':
            print("Note: Email (.eml) parsing is not fully implemented in this demo. Skipping.")
            return [] # Placeholder: You'd need a dedicated email parser here
        else:
            print(f"Unsupported document type: {content_type} / {file_extension}. Skipping.")
            return []

        # ENHANCED: Use smart splitter
        splitter = create_smart_splitter()
        chunks = splitter.split_documents(docs)
        print(f"Processed into {len(chunks)} chunks using enhanced splitter.")

        # Clean up temporary file
        os.remove(temp_file_path)
        return chunks

    except requests.exceptions.RequestException as req_err:
        print(f"Error downloading document from {document_url}: {req_err}")
    except Exception as e:
        print(f"Error processing document from {document_url}: {e}")
        traceback.print_exc()
    return []

# --- 2. Pinecone Setup (Dynamic per run for given document URL) ---
def setup_pinecone_for_url(document_url_hash: str, chunks_to_upsert: List[Document]):
    global index, vectorstore, pc, index_name, embedding_model, sparse_encoder

    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist. Creating it...")
        try:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
            )
            print(f"Index '{index_name}' created. Waiting for it to be ready...")
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print("Pinecone index is ready.")
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            raise # Re-raise to stop execution

    index = pc.Index(index_name)

    print(f"\n--- Upserting {len(chunks_to_upsert)} chunks for document URL hash '{document_url_hash}' ---")
    BATCH_SIZE = 100
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks_to_upsert):
        dense_vector = embedding_model.embed_query(chunk.page_content)
        sparse_vector_data = sparse_encoder.encode_queries([chunk.page_content])[0]

        metadata = chunk.metadata.copy()
        # Ensure numerical types are cast correctly for Pinecone
        if 'start_index' in metadata:
            metadata['start_index'] = int(metadata['start_index'])
        if 'page' in metadata:
            metadata['page'] = int(metadata['page'])
        metadata['text'] = chunk.page_content # Store full text in metadata
        metadata['document_url_hash'] = document_url_hash # Tag with the unique ID for filtering

        file_name = os.path.basename(chunk.metadata.get('source', 'unknown')).replace('.', '-')
        page_num = chunk.metadata.get('page', 0)
        vector_id = f"doc_{document_url_hash}_chunk_{i}-{file_name}-{page_num}" # Unique ID per chunk

        vectors_to_upsert.append({
            "id": vector_id,
            "values": dense_vector,
            "sparse_values": sparse_vector_data,
            "metadata": metadata
        })

    try:
        for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
            batch = vectors_to_upsert[i : i + BATCH_SIZE]
            index.upsert(vectors=batch)
        print(f"Successfully added chunks for document URL hash '{document_url_hash}'.")
        print(f"Current index stats: {index.describe_index_stats()}")
    except Exception as e:
        print(f"\n❌ An error occurred during upserting documents for URL hash '{document_url_hash}':")
        traceback.print_exc()
        raise # Re-raise the exception

    # Initialize retriever after upsert
    vectorstore = PineconeHybridSearchRetriever(
        embeddings=embedding_model,
        sparse_encoder=sparse_encoder,
        index=index,
        text_key="text",
    )
    print("PineconeHybridSearchRetriever initialized/updated.")

# --- 3. Enhanced Hybrid Search Function ---
def perform_hybrid_search(search_query_item, k: int = 8, document_filter: Optional[dict] = None) -> List[Document]:
    """
    Performs a hybrid search on the Pinecone index using the provided query item.
    ENHANCED: Increased default k from 6 to 8 for better coverage
    """
    query_text = search_query_item.query
    search_type = search_query_item.type

    try:
        retrieved_docs = vectorstore.invoke(
            query_text,
            config={"configurable": {"search_kwargs": {"k": k, "filter": document_filter}}}
        )
        for doc in retrieved_docs:
            doc.metadata['source_file'] = doc.metadata.get('source', 'N/A')
            doc.metadata['source_page'] = doc.metadata.get('page', 'N/A')
        return retrieved_docs
    except Exception as e:
        print(f"❌ Error during hybrid search for query '{query_text}' with filter {document_filter}: {e}")
        traceback.print_exc()
        return []

# --- RE-SEARCH LOGIC FOR FAILED ANSWERS ---
def re_search_with_different_strategy(question: str, failed_answer: str, document_filter: Optional[dict] = None) -> List[Document]:
    """Re-search with different strategy when first attempt fails"""
    
    if failed_answer == "Information not found":
        print(f"Re-searching for question: {question}")
        # Try broader, more general searches
        main_topic = extract_main_topic(question)
        
        broad_searches = [
            InternalSearchQueryItem(
                type="semantic",
                query=f"policy terms related to {main_topic}",
                reason="Broader search for related policy terms"
            ),
            InternalSearchQueryItem(
                type="keyword",
                query=f"{main_topic} policy document",
                reason="General policy document search"
            ),
            InternalSearchQueryItem(
                type="semantic",
                query=f"insurance coverage {main_topic}",
                reason="Insurance coverage search"
            )
        ]
        
        all_docs = []
        for search in broad_searches:
            docs = perform_hybrid_search(search, k=10, document_filter=document_filter)  # Increase k for re-search
            all_docs.extend(docs)
        
        # Remove duplicates
        unique_docs = []
        seen_keys = set()
        for doc in all_docs:
            doc_key = (doc.metadata.get('source', 'N/A'), doc.metadata.get('page', 'N/A'), doc.page_content)
            if doc_key not in seen_keys:
                unique_docs.append(doc)
                seen_keys.add(doc_key)
        
        return unique_docs
    
    return []

print("\n--- Enhanced Hybrid Search Function (Ready with re-search capability) ---")

# --- 4. Enhanced Planner Agent Code ---
class InternalSearchQueryItem(BaseModel):
    """A single internal document search query, its type, and the reason for it."""
    type: Literal["keyword", "semantic"] = Field(description="Type of query: 'keyword' for keyword/BM25 search or 'semantic' for semantic/vector search.")
    query: str = Field(description="The specific search term or phrase to use for an internal document search (e.g., in Pinecone).")
    reason: str = Field(description="Your reasoning for why this internal document search is important to fully answer the original query.")

class InternalSearchPlan(BaseModel):
    """A comprehensive plan for internal document searches to best answer a given query."""
    searches: List[InternalSearchQueryItem] = Field(
        description=(
            "A list of distinct and comprehensive internal document search queries to perform. "
            "Generate at least 3 and up to 5 relevant searches, approaching the topic from various angles "
            "(e.g., specific terms, broader concepts, related policies, exclusions)."
        )
    )

parser_internal_search = PydanticOutputParser(pydantic_object=InternalSearchPlan)

# --- ENHANCED SEARCH PLAN GENERATION ---
def generate_enhanced_search_plan(question: str) -> List[InternalSearchQueryItem]:
    """Enhanced search plan generation with question-type awareness"""
    
    # Detect question type
    question_lower = question.lower()
    question_type = detect_question_type(question_lower)
    key_terms = extract_key_terms(question)
    
    searches = []
    
    if question_type == "numerical_limits":
        # For questions about limits, percentages, amounts
        searches.extend([
            InternalSearchQueryItem(
                type="keyword",
                query=f"limit {' '.join(key_terms)}",
                reason="Search for specific limits"
            ),
            InternalSearchQueryItem(
                type="keyword", 
                query=f"percentage % {' '.join(key_terms)}",
                reason="Search for percentage-based limits"
            ),
            InternalSearchQueryItem(
                type="semantic",
                query=f"What are the maximum charges for {' '.join(key_terms)}",
                reason="Semantic search for charge limits"
            ),
            InternalSearchQueryItem(
                type="keyword",
                query=f"sub-limit sublimit {' '.join(key_terms)}",
                reason="Search for sub-limits"
            )
        ])
    
    elif question_type == "coverage_conditions":
        # For questions about what's covered and conditions
        searches.extend([
            InternalSearchQueryItem(
                type="keyword",
                query=f"covered {' '.join(key_terms)} conditions",
                reason="Search for coverage and conditions"
            ),
            InternalSearchQueryItem(
                type="keyword",
                query=f"{' '.join(key_terms)} exclusions limitations",
                reason="Search for exclusions and limitations"
            ),
            InternalSearchQueryItem(
                type="semantic",
                query=f"Under what circumstances is {' '.join(key_terms)} covered",
                reason="Semantic search for coverage circumstances"
            )
        ])
    
    elif question_type == "time_based":
        # For waiting periods, grace periods, etc.
        searches.extend([
            InternalSearchQueryItem(
                type="keyword",
                query=f"waiting period {' '.join(key_terms)}",
                reason="Search for waiting periods"
            ),
            InternalSearchQueryItem(
                type="keyword",
                query=f"grace period {' '.join(key_terms)}",
                reason="Search for grace periods"
            ),
            InternalSearchQueryItem(
                type="semantic",
                query=f"How long before {' '.join(key_terms)} is effective",
                reason="Semantic search for time requirements"
            )
        ])
    
    elif question_type == "definition":
        # For definition questions
        searches.extend([
            InternalSearchQueryItem(
                type="keyword",
                query=f"definition {' '.join(key_terms)}",
                reason="Search for definitions"
            ),
            InternalSearchQueryItem(
                type="semantic",
                query=f"What does {' '.join(key_terms)} mean",
                reason="Semantic search for meaning"
            ),
            InternalSearchQueryItem(
                type="keyword",
                query=f"{' '.join(key_terms)} means",
                reason="Search for meaning clauses"
            )
        ])
    
    # Add general searches
    searches.extend([
        InternalSearchQueryItem(
            type="keyword",
            query=' '.join(key_terms),
            reason="Direct keyword search"
        ),
        InternalSearchQueryItem(
            type="semantic",
            query=question,
            reason="Full question semantic search"
        )
    ])
    
    return searches[:5]  # Limit to 5 searches

# --- ROBUST PARSING FUNCTION ---
def robust_parse_search_plan(raw_response: str, question: str = "") -> Optional[InternalSearchPlan]:
    """
    Attempts to parse the search plan with multiple fallback strategies.
    ENHANCED: Uses question-aware fallback
    """
    try:
        # First, try standard Pydantic parsing
        return parser_internal_search.parse(raw_response)
    except Exception as e1:
        print(f"Standard parsing failed: {e1}")
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return InternalSearchPlan(**data)
        except Exception as e2:
            print(f"JSON extraction failed: {e2}")
        
        try:
            # ENHANCED: Use question-aware fallback
            print("Creating enhanced fallback search plan...")
            fallback_searches = generate_enhanced_search_plan(question)
            return InternalSearchPlan(searches=fallback_searches)
        except Exception as e3:
            print(f"Enhanced fallback creation failed: {e3}")
            # Final fallback
            return InternalSearchPlan(searches=[
                InternalSearchQueryItem(
                    type="keyword",
                    query="policy terms conditions",
                    reason="General search for policy information"
                ),
                InternalSearchQueryItem(
                    type="semantic", 
                    query="coverage benefits exclusions",
                    reason="Semantic search for coverage details"
                )
            ])

# --- ENHANCED PROMPT FOR SEARCH PLANNING ---
prompt_internal_search_planner_enhanced = ChatPromptTemplate.from_messages([
    ("system",
    """You are an expert internal document search planner for insurance policy documents.
Your task is to analyze a user's natural language query and generate up to five diverse, effective search queries for retrieving relevant clauses, rules, or information from a policy document database (e.g. Pinecone).

IMPORTANT: You MUST respond with valid JSON format matching this exact structure:

{{
  "searches": [
    {{
      "type": "keyword",
      "query": "your search query here",
      "reason": "explanation for this search"
    }}
  ]
}}

ENHANCED INSTRUCTIONS FOR SPECIFIC QUESTION TYPES:

For NUMERICAL/LIMIT questions (sub-limits, percentages, amounts):
- Include searches for "limit", "percentage", "%", "maximum", "minimum"
- Search for specific numerical patterns
- Include terms like "sub-limit", "capped at", "up to"

For COVERAGE questions (what's covered, conditions):
- Search for "covered", "coverage", "benefits", "included"
- Also search for "exclusions", "limitations", "conditions"
- Include specific medical/insurance terms

For TIME-BASED questions (waiting periods, grace periods):
- Search for "waiting period", "grace period", "effective date"
- Include time-related terms like "months", "days", "years"
- Search for "continuous coverage" requirements

For DEFINITION questions:
- Search for "definition", "means", "shall mean"
- Include the specific term being defined
- Search for regulatory references

Instructions:
- Generate a total of no more than 5 search queries per request.
- Ensure diverse query types:
    - At least 2 keyword-based queries (for keyword/BM25 search)
    - At least 2 semantically-rich natural language queries (for semantic/vector search)
    - Vary specificity: include both broad and focused queries.
- Consider aspects such as:
    - Key terms, synonyms, related concepts
    - Policy types or categories
    - Clauses, exclusions, conditions, procedures
    - Headings/sections (e.g., "Coverage", "Exclusions")
    - Demographics, regions, or medical terms if relevant
    - Variations on exclusions or limitations if applicable
    - Numerical patterns and limits
- Avoid redundant queries; make each target a different aspect or wording.
- Output as a list, each item with: type ("keyword" or "semantic"), query, and a one-sentence reason for inclusion.

Do not generate more than 5 queries total.
"""),
    ("human", "User query: {input}")
])

model_internal_search = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
internal_search_planner_chain_enhanced = prompt_internal_search_planner_enhanced | model_internal_search

print("\n--- Enhanced Planner Agent (Ready) ---")

# --- 5. Enhanced Answer Synthesis Agent Code ---
class AnswerItem(BaseModel):
    """A single answer to a specific question based on retrieved documents.
    This intermediate model is useful for LLM's internal structured output,
    but the final HackRx API response requires a simple list of strings.
    """
    answer: str = Field(description="The concise, factual answer derived from the retrieved documents. State 'Information not found' if the answer cannot be determined from the provided context.")

class AnswersResponse(BaseModel):
    """A list of answers corresponding to the provided questions.
    This model is specifically designed to match the HackRx API output format.
    """
    answers: List[str] = Field(description="A list of generated answers, each corresponding to a question in the input list.")

parser_answers = PydanticOutputParser(pydantic_object=AnswersResponse)

# --- ANSWER VALIDATION ---
def validate_answer_completeness(question: str, answer: str, context: str) -> bool:
    """Validate if answer adequately addresses the question"""
    if answer == "Information not found":
        # Check if context actually contains relevant information
        key_terms = extract_key_terms(question)
        if any(term.lower() in context.lower() for term in key_terms):
            return False  # Context has info but answer says not found
    
    # Check for vague answers when specific info might be available
    vague_patterns = ["with certain conditions", "but with restrictions", "under specific circumstances"]
    if any(pattern in answer.lower() for pattern in vague_patterns):
        # Look for specific conditions in context
        if re.search(r'\d+%|\d+ days|\d+ months|specific.*act|comply.*regulation', context, re.IGNORECASE):
            return False  # Context has specific info but answer is vague
    
    return True

# --- ROBUST ANSWER PARSING FUNCTION ---
def robust_parse_answer(raw_response: str) -> str:
    """
    Attempts to parse the answer with multiple fallback strategies.
    ENHANCED: Better pattern matching and validation
    """
    print(f"Raw response to parse: {raw_response[:200]}...")  # Debug log
    
    try:
        # First, try standard Pydantic parsing
        answer_response = parser_answers.parse(raw_response)
        if answer_response and answer_response.answers and len(answer_response.answers) > 0:
            result = answer_response.answers[0].strip()
            print(f"Successfully parsed with Pydantic: {result[:100]}...")
            return result
    except Exception as e1:
        print(f"Standard answer parsing failed: {e1}")
    
    try:
        # Try to extract JSON from the response - be more precise
        json_match = re.search(r'\{[^{}]*"answers"[^{}]*\[[^\]]*\][^{}]*\}', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            print(f"Found JSON match: {json_str}")
            data = json.loads(json_str)
            if 'answers' in data and isinstance(data['answers'], list) and len(data['answers']) > 0:
                result = str(data['answers'][0]).strip()
                print(f"Successfully extracted from JSON: {result[:100]}...")
                return result
    except Exception as e2:
        print(f"JSON answer extraction failed: {e2}")
    
    # If all parsing fails, try to extract meaningful content from raw response
    try:
        # Clean up the response
        cleaned_response = raw_response.strip()
        
        # Remove JSON artifacts if present but malformed
        if cleaned_response.startswith('{') and '"answers"' in cleaned_response:
            # Try to extract just the content between quotes
            start_pattern = r'^\{.*?"answers".*?\[.*?"'
            end_pattern = r'".*?\].*?\}$'
            cleaned_response = re.sub(start_pattern, '', cleaned_response)
            cleaned_response = re.sub(end_pattern, '', cleaned_response)
        cleaned_response = cleaned_response.strip().strip('"')
        
        # Look for common patterns that indicate the actual answer
        patterns = [
            r'"answer":\s*"([^"]*)"',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'Response:\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
            if match:
                result = match.group(1).strip()
                print(f"Pattern matched: {result[:100]}...")
                return result
        
        # If no patterns match, return the cleaned response if it's reasonable
        if (len(cleaned_response) < 1000 and 
            len(cleaned_response) > 5 and 
            cleaned_response.lower() != "information not found" and
            not cleaned_response.startswith('{')):
            print(f"Using cleaned response: {cleaned_response[:100]}...")
            return cleaned_response
            
    except Exception as e3:
        print(f"Fallback answer extraction failed: {e3}")
    
    print("All parsing methods failed, returning default")
    return "Information not found"

# --- ENHANCED ANSWER SYNTHESIS PROMPT ---
prompt_answer_synthesis_enhanced = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert insurance policy assistant. Your task is to answer the user's question concisely and accurately, based *only* on the provided context documents.

CRITICAL INSTRUCTIONS FOR SPECIFIC QUESTION TYPES:
- For coverage questions: Always specify WHAT is covered, UNDER WHAT CONDITIONS, and any LIMITATIONS
- For numerical limits: Always extract specific percentages, amounts, or time periods
- For conditions: List specific requirements clearly
- For exclusions: Mention key exclusions if present in context
- For waiting periods: Specify exact time periods (days, months, years)
- For definitions: Provide complete definitions with all criteria

IMPORTANT: You MUST respond with valid JSON format matching this exact structure:

{{
  "answers": ["your detailed answer here"]
}}

Examples of good vs bad answers:
❌ BAD: "Yes, but with certain conditions"
✅ GOOD: "Yes, medical expenses for organ donors are covered when the organ is donated to an insured person and complies with the Transplantation of Human Organs Act, 1994"

❌ BAD: "Information not found" (when numerical data exists in context)
✅ GOOD: "Yes, room rent is capped at 1% of Sum Insured and ICU charges at 2% of Sum Insured for Plan A"

❌ BAD: "There is a waiting period"
✅ GOOD: "The waiting period is 36 months of continuous coverage"

EXTRACTION PRIORITIES:
1. Look for specific numbers, percentages, amounts, time periods
2. Extract exact conditions and requirements
3. Include relevant act names, regulations, or legal references
4. Mention specific exclusions or limitations
5. Provide complete definitions with all criteria

Guidelines:
- If the answer is not explicitly available in the provided context, state "Information not found"
- Do NOT make up information or infer beyond the given context
- Extract the most relevant information and present it directly
- Do not add conversational filler or introductory phrases
- Ensure the answer is directly responsive to the question
- When the question asks about 'coverage', 'limits', 'conditions', 'benefits', 'waiting periods', 'exclusions', or similar quantitative/conditional aspects, focus on those specific details rather than just providing definitions

Context documents:
{context}
"""),
    ("human", "Question to answer: {question_text}")
])
 
model_answer_synthesis = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
answer_synthesis_chain_enhanced = prompt_answer_synthesis_enhanced | model_answer_synthesis

print("\n--- Enhanced Answer Synthesis Agent (Ready) ---")

# --- 6. Enhanced Overall Orchestration Logic ---

def run_enhanced_hackrx_pipeline(questions: List[str], documents_url: Optional[str] = None):
    """
    ENHANCED: Runs the full RAG pipeline with improvements for better accuracy
    """
    print(f"\n===== Running ENHANCED HackRx Pipeline for {len(questions)} Questions =====")

    current_document_hash = None # Initialize to None
    document_filter = None # Initialize filter to None

    if documents_url:
        current_document_hash = hashlib.md5(documents_url.encode('utf-8')).hexdigest()
        print(f"Processing documents from URL: {documents_url} (Hash: {current_document_hash})")

        print("--- Step A: Dynamic Document Ingestion ---")
        try:
            document_chunks = download_and_process_document(documents_url)
            if not document_chunks:
                print(f"No documents processed from URL: {documents_url}. Cannot answer questions.")
                return {"answers": ["Information not found."] * len(questions)}

            print("--- Step B: Upserting chunks to Pinecone ---")
            setup_pinecone_for_url(current_document_hash, document_chunks)

            # Ensure the global vectorstore is updated
            global vectorstore
            vectorstore = PineconeHybridSearchRetriever(
                embeddings=embedding_model,
                sparse_encoder=sparse_encoder,
                index=index,
                text_key="text",
            )
            # Define the filter AFTER a successful document ingestion
            document_filter = {"document_url_hash": current_document_hash}

        except Exception as e:
            print(f"❌ Critical error during document ingestion/Pinecone setup for URL '{documents_url}': {e}")
            traceback.print_exc()
            return {"answers": ["Critical Error: Document ingestion failed."] * len(questions)}
    else:
        print("No 'documents' URL provided. This pipeline requires a document context.")
        return {"answers": ["Critical Error: No document context provided."] * len(questions)}

    final_answers_list = []

    for q_idx, question in enumerate(questions):
        print(f"\n--- Processing Question {q_idx + 1}/{len(questions)}: '{question}' ---")
        print(f"Question type detected: {detect_question_type(question)}")

        # Step 1: Use the Enhanced Planner Agent to get a search plan for the current question
        print("--- Step 1: Running Enhanced Planner Agent to create search plan ---")
        plan = None # Initialize plan
        try:
            raw_planner_response = internal_search_planner_chain_enhanced.invoke({
                "input": question,
            })
            plan = robust_parse_search_plan(raw_planner_response.content, question)
            
            if plan and plan.searches:
                print("Enhanced Planner Agent generated the following search plan:")
                for item in plan.searches:
                    print(f"  - Type: {item.type}, Query: '{item.query}' (Reason: {item.reason[:70]}...)")
            else:
                print("❌ Failed to generate valid search plan")
        except Exception as e:
            print(f"❌ Error running Enhanced Planner Agent for question '{question}': {e}")
            traceback.print_exc()

        all_retrieved_documents = []
        retrieved_doc_unique_keys = set()

        # Step 2: Use the Enhanced Hybrid Search Function for each planned query
        if plan and plan.searches:
            print("\n--- Step 2: Executing Enhanced Hybrid Search for each planned query ---")
            for search_item in plan.searches:
                docs = perform_hybrid_search(search_query_item=search_item, k=8, document_filter=document_filter) # k=8 for enhanced coverage
                for doc in docs:
                    doc_key = (doc.metadata.get('source', 'N/A'), doc.metadata.get('page', 'N/A'), doc.page_content)
                    if doc_key not in retrieved_doc_unique_keys:
                        all_retrieved_documents.append(doc)
                        retrieved_doc_unique_keys.add(doc_key)
            print(f"\n--- Enhanced Hybrid Search completed. Total unique documents retrieved: {len(all_retrieved_documents)} ---")

            if not all_retrieved_documents:
                print("No documents retrieved for this question using the specified document context.")
                final_answers_list.append("Information not found")
                continue

            # Prepare context for the Enhanced Answer Synthesis Agent
            context_str = "\n\n".join([doc.page_content for doc in all_retrieved_documents])

            # Step 3: Use the Enhanced Answer Synthesis Agent to generate the answer
            print("\n--- Step 3: Running Enhanced Answer Synthesis Agent ---")
            raw_llm_response = None # Initialize raw_llm_response
            try:
                raw_llm_response = answer_synthesis_chain_enhanced.invoke({
                    "context": context_str,
                    "question_text": question,
                })
                
                synthesized_answer = robust_parse_answer(raw_llm_response.content)
                print(f"  Synthesized Answer: {synthesized_answer}")
                
                # ENHANCED: Validate answer completeness
                if not validate_answer_completeness(question, synthesized_answer, context_str):
                    print("Answer validation failed. Attempting re-search...")
                    
                    # Step 4: Re-search with different strategy if answer is inadequate
                    additional_docs = re_search_with_different_strategy(question, synthesized_answer, document_filter)
                    if additional_docs:
                        print(f"Re-search found {len(additional_docs)} additional documents")
                        # Combine original and additional documents
                        all_docs_combined = all_retrieved_documents + additional_docs
                        enhanced_context = "\n\n".join([doc.page_content for doc in all_docs_combined])
                        
                        # Try synthesis again with enhanced context
                        try:
                            raw_llm_response_retry = answer_synthesis_chain_enhanced.invoke({
                                "context": enhanced_context,
                                "question_text": question,
                            })
                            retry_answer = robust_parse_answer(raw_llm_response_retry.content)
                            if retry_answer != "Information not found":
                                synthesized_answer = retry_answer
                                print(f"  Enhanced Answer after re-search: {synthesized_answer}")
                        except Exception as retry_e:
                            print(f"Re-search synthesis failed: {retry_e}")
                
                # Ensure we only add one answer per question
                if synthesized_answer and synthesized_answer.strip():
                    final_answers_list.append(synthesized_answer.strip())
                else:
                    final_answers_list.append("Information not found")

            except Exception as e:
                print(f"❌ Error during Enhanced Answer Synthesis for question '{question}': {e}")
                print(f"Raw LLM response (if available): {raw_llm_response.content if raw_llm_response else 'N/A'}")
                traceback.print_exc()
                final_answers_list.append("Information not found")
                continue
        else:
            print("No search plan generated or plan is empty for this question. Skipping hybrid search and synthesis.")
            final_answers_list.append("Information not found")
            continue

    # Ensure we have exactly the same number of answers as questions
    while len(final_answers_list) < len(questions):
        final_answers_list.append("Information not found")
    
    # Trim to exact number of questions if somehow we have more
    final_answers_list = final_answers_list[:len(questions)]

    hackrx_response = {"answers": final_answers_list}
    print(f"\nFinal response validation: {len(questions)} questions, {len(final_answers_list)} answers")
    return hackrx_response

# --- TEST CASES FOR VALIDATION ---
test_cases = [
    {
        "question": "Are there any sub-limits on room rent and ICU charges for Plan A?",
        "expected_keywords": ["1%", "2%", "Sum Insured", "room rent", "ICU"],
        "should_not_be": "Information not found"
    },
    {
        "question": "Are the medical expenses for an organ donor covered under this policy?",
        "expected_keywords": ["Transplantation of Human Organs Act", "insured person", "donor"],
        "should_not_be": "certain conditions and exclusions"
    },
    {
        "question": "Does this policy cover maternity expenses, and what are the conditions?",
        "expected_keywords": ["24 months", "two deliveries", "terminations"],
        "should_include_all": True
    }
]

def validate_test_cases(results: dict):
    """Validate results against test cases"""
    print("\n--- VALIDATION RESULTS ---")
    for i, test_case in enumerate(test_cases):
        if i < len(results["answers"]):
            answer = results["answers"][i]
            question = test_case["question"]
            
            print(f"\nTest Case {i+1}: {question}")
            print(f"Answer: {answer}")
            
            # Check if answer should not be a specific value
            if "should_not_be" in test_case and answer == test_case["should_not_be"]:
                print(f"❌ FAIL: Answer should not be '{test_case['should_not_be']}'")
            else:
                print("✅ PASS: Answer is not the forbidden value")
            
            # Check for expected keywords
            if "expected_keywords" in test_case:
                found_keywords = [kw for kw in test_case["expected_keywords"] if kw.lower() in answer.lower()]
                print(f"Expected keywords found: {found_keywords}")
                if len(found_keywords) > 0:
                    print("✅ PASS: Some expected keywords found")
                else:
                    print("❌ FAIL: No expected keywords found")
