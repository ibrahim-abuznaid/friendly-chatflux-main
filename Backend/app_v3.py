import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import openai
from langchain_core.documents import Document
from typing import List, TypedDict, Optional
import re
import argparse
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8080", "http://127.0.0.1:8080"],
        "methods": ["POST", "GET", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load environment variables
load_dotenv()

# Set up LangSmith environment variables
os.environ["LANGSMITH_TRACING"] = "true"

# Create OpenAI client
openai_client = wrap_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Modify the embeddings setup to use the correct Gemini embeddings configuration
def get_embeddings(llm_model="openai"):

    """Get embeddings based on specified model"""
    if llm_model == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif llm_model == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",  # Correct model name for Gemini embeddings
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type="retrieval_query",
            dimension=768  # Specify embedding dimension
        )
    else:
        raise ValueError(f"Unsupported LLM model for embeddings: {llm_model}")

# Modify the get_llm function to use the correct Gemini model name
def get_llm(llm_model="openai"):
    """Get LLM based on specified model"""
    if llm_model == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif llm_model == "gemini":

        return ChatGoogleGenerativeAI(
            model="gemini-2.0-pro-exp-02-05",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model}")

# Initialize global variables
llm = get_llm("openai")
embeddings = get_embeddings("openai")

# Define structured output for decomposed queries
class DecomposedQuery(TypedDict):
    main_query: str
    sub_queries: List[str]
    filters: Optional[dict]

# Add this new class near the top with other imports
class Message(TypedDict):
    role: str  # "user" or "assistant" 
    content: str

# Basic Query Translation
@traceable(name="basic_query_translation")
def basic_translate_query(query):
    """Basic query translation"""
    translation_prompt = ChatPromptTemplate.from_template(
        """Rewrite the following query to make it more clear and specific for a search: {query}"""
    )
    translation_chain = translation_prompt | llm
    translated_query = translation_chain.invoke({"query": query}).content
    return translated_query

# RAG Fusion Query Generation
@traceable(name="generate_queries")
def generate_queries(question: str, n: int = 3) -> List[str]:
    """Generate multiple versions of the input question for RAG Fusion"""
    prompt = ChatPromptTemplate.from_template(
        """Generate {n} different versions of the given question. 
        Make the questions more specific and use different wordings.
        Return only the questions, one per line.
        
        Question: {question}
        
        Different versions:"""
    )
    
    response = llm.invoke(
        prompt.format_messages(question=question, n=n)
    )
    
    queries = [q.strip() for q in response.content.split('\n') if q.strip()]
    return queries[:n]

# Query Decomposition
@traceable(name="query_decomposition")
def decompose_query(query: str) -> DecomposedQuery:
    """Decompose complex queries into simpler components"""
    # ... existing decompose_query implementation ...
    # (Keep the full implementation from Decomposition-app.py)
    decomposition_prompt = ChatPromptTemplate.from_template(
        """You are an expert at breaking down complex queries about building codes and regulations into simpler components.
        
        Follow these steps to decompose the query:
        1. First, identify the main topic or requirement being asked about
        2. Break down complex questions into simpler sub-queries
        3. Identify any specific filters (e.g., dates, locations, specific code sections)
        4. Use Chain of Thought reasoning to explain your decomposition
        
        Guidelines:
        - Create sub-queries that are specific and focused
        - Maintain the original intent of the question
        - Identify any implicit requirements
        
        Query: {query}
        
        Think through this step by step:
        1) Main topic analysis:
        2) Breaking down into components:
        3) Identifying filters:
        4) Final decomposition:
        
        Return the decomposed query in this format:
        Main Query: [The primary question]
        Sub Queries:
        - [Sub query 1]
        - [Sub query 2]
        Filters: [Any specific filters identified]
        """
    )
    
    response = llm.invoke(decomposition_prompt.format(query=query))
    
    try:
        lines = response.content.split('\n')
        main_query = ""
        sub_queries = []
        filters = {}
        
        for line in lines:
            if line.startswith("Main Query:"):
                main_query = line.replace("Main Query:", "").strip()
            elif line.startswith("-"):
                sub_queries.append(line.replace("-", "").strip())
            elif line.startswith("Filters:"):
                filters_str = line.replace("Filters:", "").strip()
                if filters_str and filters_str.lower() != "none":
                    filters = {f.split(':')[0].strip(): f.split(':')[1].strip() 
                             for f in filters_str.split(',')}
        
        return DecomposedQuery(
            main_query=main_query,
            sub_queries=sub_queries,
            filters=filters
        )
    except Exception as e:
        print(f"Error parsing decomposition: {str(e)}")
        return DecomposedQuery(
            main_query=query,
            sub_queries=[query],
            filters={}
        )

# Reciprocal Rank Fusion
@traceable(name="rank_fusion")
def reciprocal_rank_fusion(doc_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """Combine multiple document rankings using RRF"""
    doc_scores = {}
    
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            doc_id = f"{doc.metadata.get('document_id')}_{doc.metadata.get('page_number')}"
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += 1 / (k + rank + 1)
    
    doc_map = {}
    for doc_list in doc_lists:
        for doc in doc_list:
            doc_id = f"{doc.metadata.get('document_id')}_{doc.metadata.get('page_number')}"
            doc_map[doc_id] = doc
    
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_doc_ids]

# Document Preparation
@traceable(name="document_preparation")
def prepare_documents(documents):
    """Split documents into chunks with tracing and add metadata"""
    if not documents:
        raise ValueError("No input documents provided to prepare_documents")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    

    processed_docs = []
    last_seen_page_number = None
    
    for doc_idx, doc in enumerate(documents):
        if not doc.page_content.strip():
            print(f"Warning: Empty document at index {doc_idx}")
            continue
            
        splits = text_splitter.split_documents([doc])
        
        if not splits:
            print(f"Warning: No splits generated for document at index {doc_idx}")
            continue
            
        for chunk_idx, split_doc in enumerate(splits):
            page_match = re.search(r'2500-(\d+)\n---', split_doc.page_content)
            if page_match:
                page_number = int(page_match.group(1))
                last_seen_page_number = page_number
            else:
                if last_seen_page_number is not None:
                    page_number = last_seen_page_number + 1
                    last_seen_page_number = page_number
                else:
                    page_number = 1
                    last_seen_page_number = 1
            
            split_doc.metadata.update({
                'document_id': f'doc_{doc_idx}',
                'page_number': page_number
            })
            processed_docs.append(split_doc)
    
    if not processed_docs:
        raise ValueError("No documents were successfully processed")
        
    return processed_docs

# Vector Store Operations
@traceable(name="vector_store_creation")
def create_vector_store(document_splits, llm_model="openai"):
    """Create a vector store from document splits with tracing"""
    try:
        if not document_splits:
            raise ValueError("No documents provided for vector store creation")
            
        # Get the appropriate embeddings for the model
        current_embeddings = get_embeddings(llm_model)
        
        vectorstore = FAISS.from_documents(
            documents=document_splits, 
            embedding=current_embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def save_vectorstore(vectorstore, path="faiss_index"):
    """Save the FAISS index to disk"""
    vectorstore.save_local(path)

def load_vectorstore(path="faiss_index"):
    """Load the FAISS index from disk"""
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings)
    return None

# Document Retrieval based on technique
@traceable(name="document_retrieval")
def retrieve_documents(vectorstore, query, technique="basic"):
    """Retrieve documents using specified technique"""
    if technique == "fusion":
        queries = generate_queries(query)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
        doc_lists = [retriever.invoke(q) for q in queries]
        fused_docs = reciprocal_rank_fusion(doc_lists)
        docs = fused_docs[:4]
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
        docs = retriever.invoke(query)
    
    return [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata)
        }
        for doc in docs
    ]

# Create RAG Chain
@traceable(name="rag_chain_creation")
def create_rag_chain(vectorstore, conversation_history: List[Message] = None):
    """Create the RAG chain with conversation history"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Update prompt to include conversation history
    rag_prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant specializing in building codes and regulations. Your task is to provide accurate, clear answers based on the provided context and conversation history.

        Previous conversation:
        {conversation_history}

        Guidelines for your response:
        1. Base your answer on the context and previous conversation
        2. For each specific requirement or regulation you mention, cite the page number (2500-xxx) in parentheses at the end of that statement
        3. Only include page numbers that are explicitly marked in the text (following the format "2500-xxx---")
        4. If a section doesn't have an explicit page marker, don't cite a page number for that information
        5. Present information in a clear, organized manner
        6. If the context doesn't contain enough information to fully answer the question, acknowledge this

        Context: {context}

        Question: {question}

        Please provide your detailed response following the guidelines above:"""
    )
    
    # Format conversation history
    formatted_history = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in (conversation_history or [])
    ])
    
    return (
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "conversation_history": lambda _: formatted_history
        } 
        | rag_prompt 
        | llm
    )

# Add this after the existing query translation functions

@traceable(name="hyde_query_translation")
def hyde_translate_query(query: str, llm_model: str = "openai") -> tuple[str, str]:
    """Generate hypothetical document for better retrieval using HyDE technique"""
    # Get the appropriate LLM for HyDE
    hyde_llm = get_llm(llm_model)
    
    hyde_prompt = ChatPromptTemplate.from_template(
        """Please write a scientific paper passage to answer the question
        Question: {question}
        Passage:"""
    )
    
    # Generate hypothetical document using specified model
    generate_hyde_doc = (
        hyde_prompt 
        | hyde_llm 
        | StrOutputParser()
    )
    
    try:
        hypothetical_doc = generate_hyde_doc.invoke({"question": query})
        return query, hypothetical_doc
    except Exception as e:
        raise Exception(f"Failed to generate HyDE document with {llm_model}: {str(e)}")

# Main RAG Pipeline
@traceable(name="rag_pipeline")
def process_query(user_query, qtt="basic", llm_model="openai", conversation_history: List[Message] = None):
    """Complete RAG pipeline with conversation history"""
    try:
        # Update LLM and embeddings based on specified model
        global llm, embeddings
        llm = get_llm(llm_model)
        embeddings = get_embeddings(llm_model)

        # Load saved vector store with allow_dangerous_deserialization=True
        vectorstore = None
        try:
            vectorstore = FAISS.load_local(
                "faiss_index", 
                embeddings,
                allow_dangerous_deserialization=True  # Add this parameter
            )
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            # Initialize if not found
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_path = os.path.join(script_dir, "docx.md")
            loader = TextLoader(docs_path, encoding="utf-8")
            documents = loader.load()
            document_splits = prepare_documents(documents)
            vectorstore = create_vector_store(document_splits, llm_model)
            vectorstore.save_local("faiss_index")

        # Process query based on technique
        if qtt == "basic":
            refined_query = basic_translate_query(user_query)
            extra_info = None
        elif qtt == "fusion":
            refined_query = user_query
            extra_info = generate_queries(user_query)
        elif qtt == "decomposition":
            decomposed = decompose_query(user_query)
            refined_query = decomposed["main_query"]
            extra_info = decomposed
        elif qtt == "hyde":
            refined_query, hypothetical_doc = hyde_translate_query(user_query, llm_model)
            extra_info = {"hypothetical_doc": hypothetical_doc}
            # Create a temporary Document object with the hypothetical text
            hyde_doc = Document(page_content=hypothetical_doc)
            # Get embeddings for the hypothetical document using specified model
            hyde_embedding = embeddings.embed_documents([hyde_doc.page_content])[0]
            # Use the hypothetical document embedding for retrieval
            retrieved_docs = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in vectorstore.similarity_search_by_vector(hyde_embedding, k=4)
            ]
            return {
                "original_query": user_query,
                "refined_query": refined_query,
                "hypothetical_doc": hypothetical_doc,
                "response": create_rag_chain(vectorstore, conversation_history).invoke(user_query).content,
                "retrieved_documents": retrieved_docs,
                "llm_model": llm_model,
                "conversation_history": conversation_history
            }
        else:
            raise ValueError(f"Unknown query translation technique: {qtt}")

        # Retrieve documents
        retrieved_docs = retrieve_documents(vectorstore, refined_query, 
                                         "fusion" if qtt == "fusion" else "basic")
        
        # Generate Response with conversation history
        rag_chain = create_rag_chain(vectorstore, conversation_history)
        response = rag_chain.invoke(user_query)
        
        result = {
            "original_query": user_query,
            "refined_query": refined_query,
            "response": response.content,
            "retrieved_documents": retrieved_docs,
            "conversation_history": conversation_history
        }
        
        # Add technique-specific information
        if qtt == "fusion":
            result["generated_queries"] = extra_info
        elif qtt == "decomposition":
            result["decomposed_query"] = extra_info
            
        return result
        
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        raise Exception(f"Failed to process query: {str(e)}")

# Flask routes
@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        # Add CORS headers
        if request.method == 'OPTIONS':
            response = jsonify({})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            return response

        # Validate request data
        if not request.is_json:
            print("Error: Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.json
        if not data:
            print("Error: No request data provided")
            return jsonify({"error": "No request data provided"}), 400
            
        message = data.get("message")
        qtt = data.get("qtt", "basic")
        llm_model = data.get("llm_model", "openai")
        conversation_history = data.get("conversation_history", [])
        
        print(f"Received request data: {json.dumps(data, indent=2)}")
        
        if not message:
            print("Error: No message provided")
            return jsonify({"error": "No message provided"}), 400
            
        try:
            # Validate QTT value
            if qtt not in ["basic", "fusion", "decomposition", "hyde"]:
                return jsonify({"error": f"Invalid QTT value: {qtt}"}), 400

            # Validate LLM model
            if llm_model not in ["openai", "gemini"]:
                return jsonify({"error": f"Invalid LLM model: {llm_model}"}), 400

            # Validate conversation history format
            if not isinstance(conversation_history, list):
                return jsonify({"error": "conversation_history must be an array"}), 400

            result = process_query(message, qtt, llm_model, conversation_history)
            
            # Update conversation history with the new messages
            updated_history = conversation_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": str(result.get("response", ""))}
            ]
            
            response_data = {
                "reply": str(result.get("response", "")),
                "conversation_id": str(uuid.uuid4()),
                "context": {
                    "original_query": str(result.get("original_query", "")),
                    "refined_query": str(result.get("refined_query", "")),
                    "retrieved_documents": [
                        {
                            "page_content": str(doc.get("page_content", "")),
                            "metadata": doc.get("metadata", {})
                        }
                        for doc in result.get("retrieved_documents", [])
                    ],
                    "llm_model": str(llm_model),
                    "conversation_history": updated_history
                }
            }
            
            print(f"Prepared response data: {json.dumps(response_data, indent=2)}")
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Process query error: {str(e)}")
            return jsonify({"error": f"Failed to process query: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Add initialization timeout configuration for Google AI
def initialize_google_ai():
    """Initialize Google AI configuration"""
    try:
        import google.generativeai as genai
        genai.configure(
            api_key=os.getenv("GOOGLE_API_KEY"),
            transport="rest"  # Use REST instead of gRPC to avoid timeout issues
        )
        
        # Suppress warning messages
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
        
    except Exception as e:
        print(f"Error initializing Google AI: {str(e)}")
        raise

# Modify the main block to initialize Google AI
if __name__ == "__main__":
    # Initialize Google AI if needed
    initialize_google_ai()
    
    parser = argparse.ArgumentParser(description='Enhanced RAG Query System')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--qtt', type=str, default='basic',
                      choices=['basic', 'fusion', 'decomposition', 'hyde'],
                      help='Query translation technique to use')
    parser.add_argument('--llm_model', type=str, default='openai',
                      choices=['openai', 'gemini'],
                      help='LLM model to use')
    args = parser.parse_args()

    if args.query:
        try:
            result = process_query(args.query, args.qtt, args.llm_model)
            print("\nQuery Results:")
            print(f"\nUsing LLM Model: {args.llm_model}")
            print(f"Original Query: {result['original_query']}")
            print(f"Refined Query: {result['refined_query']}")
            
            if args.qtt == "fusion":
                print(f"\nGenerated Queries: {result['generated_queries']}")
            elif args.qtt == "decomposition":
                print("\nDecomposed Query:")
                print(json.dumps(result['decomposed_query'], indent=2))
            elif args.qtt == "hyde":
                print("\nHypothetical Document:")
                print(result['hypothetical_doc'])
                
            print(f"\nResponse: {result['response']}")
            print("\nRetrieved Documents:")
            for i, doc in enumerate(result['retrieved_documents'], 1):
                print(f"\nDocument {i}:")
                print(f"Page Number: {doc['metadata'].get('page_number', 'N/A')}")
                print(f"Content: {doc['page_content'][:200]}...")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    else:
        port = int(os.getenv("PORT", 3001))
        app.run(host="127.0.0.1", port=port, debug=True) 