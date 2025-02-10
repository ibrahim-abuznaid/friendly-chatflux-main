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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Set up LangSmith environment variables
os.environ["LANGSMITH_TRACING"] = "true"

# Create OpenAI client
openai_client = wrap_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Setup Components
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Define structured output for decomposed queries
class DecomposedQuery(TypedDict):
    main_query: str
    sub_queries: List[str]
    filters: Optional[dict]

# Task Decomposition for Query Translation
@traceable(name="query_decomposition")
def decompose_query(query: str) -> DecomposedQuery:
    """
    Decompose complex queries into simpler components using Chain of Thought approach
    """
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
    
    # Get structured decomposition from LLM
    response = llm.invoke(decomposition_prompt.format(query=query))
    
    # Parse the response into structured format
    # Note: In a production environment, you'd want more robust parsing
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
                    # Parse filters string into dictionary
                    filters = {f.split(':')[0].strip(): f.split(':')[1].strip() 
                             for f in filters_str.split(',')}
        
        return DecomposedQuery(
            main_query=main_query,
            sub_queries=sub_queries,
            filters=filters
        )
    except Exception as e:
        print(f"Error parsing decomposition: {str(e)}")
        # Fallback to original query if parsing fails
        return DecomposedQuery(
            main_query=query,
            sub_queries=[query],
            filters={}
        )

# Enhanced Query Translation with Decomposition
@traceable(name="enhanced_query_translation")
def translate_query(query: str):
    """Translate query using decomposition approach"""
    # First decompose the query
    decomposed = decompose_query(query)
    
    # Create a translation prompt that incorporates the decomposed components
    translation_prompt = ChatPromptTemplate.from_template(
        """Given this decomposed query, create a clear and specific search query.
        
        Original Query: {original_query}
        Main Topic: {main_query}
        Sub-components: {sub_queries}
        Filters: {filters}
        
        Create a comprehensive search query that incorporates all these components
        while maintaining focus on building codes and regulations.
        """
    )
    
    # Generate enhanced query
    translation_chain = translation_prompt | llm
    translated_query = translation_chain.invoke({
        "original_query": query,
        "main_query": decomposed["main_query"],
        "sub_queries": "\n- ".join(decomposed["sub_queries"]),
        "filters": json.dumps(decomposed["filters"], indent=2)
    }).content
    
    return translated_query, decomposed

# Add these utility functions before the process_query function:

@traceable(name="document_preparation")
def prepare_documents(documents):
    """Split documents into chunks with tracing and add metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400
    )
    
    processed_docs = []
    last_seen_page_number = None  # Keep track of the last assigned page number
    for doc_idx, doc in enumerate(documents):
        splits = text_splitter.split_documents([doc])
        
        for chunk_idx, split_doc in enumerate(splits):
            # Extract page number using regex - looking for the explicit markdown page marker
            page_match = re.search(r'2500-(\d+)\n---', split_doc.page_content)
            if page_match:
                # Convert the captured number to an integer and update last_seen_page_number
                page_number = int(page_match.group(1))
                last_seen_page_number = page_number
            else:
                # If no page number is found, use previous number +1 or default to 1 if none exists
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
    
    return processed_docs

@traceable(name="vector_store_creation")
def create_vector_store(document_splits):
    """Create a vector store from document splits with tracing"""
    try:
        vectorstore = FAISS.from_documents(
            documents=document_splits, 
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

@traceable(name="document_retrieval", run_type="retriever")
def retrieve_documents(vectorstore, query):
    """Retrieve most similar documents with tracing"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)
    
    formatted_docs = [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata)
        } for doc in retrieved_docs
    ]
    
    return formatted_docs

@traceable(name="rag_chain_creation")
def create_rag_chain(vectorstore):
    """Create a retrieval-augmented generation chain with tracing"""
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    
    rag_prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant specializing in building codes and regulations. Your task is to provide accurate, clear answers based solely on the provided context.

        Guidelines for your response:
        1. Base your answer ONLY on the information present in the context
        2. For each specific requirement or regulation you mention, cite the page number (2500-xxx) in parentheses at the end of that statement
        3. Only include page numbers that are explicitly marked in the text (following the format "2500-xxx---")
        4. If a section doesn't have an explicit page marker, don't cite a page number for that information
        5. Present information in a clear, organized manner
        6. If the context doesn't contain enough information to fully answer the question, acknowledge this

        Context: {context}

        Question: {question}

        Please provide your detailed response following the guidelines above:
        """
    )
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | rag_prompt 
        | llm
    )
    
    return rag_chain

# Add these functions to help persist and load the vector store
def save_vectorstore(vectorstore, path="faiss_index"):
    """Save the FAISS index to disk"""
    vectorstore.save_local(path)

def load_vectorstore(path="faiss_index"):
    """Load the FAISS index from disk"""
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings)
    return None

# Rest of the code remains similar to app.py, but with enhanced query processing
@traceable(name="rag_pipeline")
def process_query(user_query):
    """Complete RAG pipeline with decomposition-based query processing"""
    try:
        # Load documents
        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(script_dir, "docx.md")
        loader = TextLoader(docs_path, encoding="utf-8")
        documents = loader.load()

        # Enhanced Query Translation with Decomposition
        refined_query, decomposed_query = translate_query(user_query)
        
        # Document Preparation
        document_splits = prepare_documents(documents)
        
        # Vector Store Creation
        vectorstore = create_vector_store(document_splits)
        
        # Enhanced retrieval using decomposed queries
        all_retrieved_docs = []
        # Retrieve documents for main query
        main_docs = retrieve_documents(vectorstore, refined_query)
        all_retrieved_docs.extend(main_docs)
        
        # Retrieve additional documents for sub-queries
        for sub_query in decomposed_query["sub_queries"]:
            sub_docs = retrieve_documents(vectorstore, sub_query)
            all_retrieved_docs.extend(sub_docs)
        
        # Deduplicate documents based on content
        seen_contents = set()
        unique_docs = []
        for doc in all_retrieved_docs:
            if doc["page_content"] not in seen_contents:
                seen_contents.add(doc["page_content"])
                unique_docs.append(doc)
        
        # RAG Chain
        rag_chain = create_rag_chain(vectorstore)
        
        # Generate Response
        response = rag_chain.invoke(refined_query)
        
        return {
            "original_query": user_query,
            "refined_query": refined_query,
            "decomposed_query": decomposed_query,
            "response": response.content,
            "retrieved_documents": unique_docs[:4]  # Limit to top 4 most relevant docs
        }
        
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        raise Exception(f"Failed to process query: {str(e)}")

# Flask routes
@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.json
        message = data.get("message")
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
            
        result = process_query(message)
        
        return jsonify({
            "reply": result["response"],
            "context": {
                "original_query": result["original_query"],
                "refined_query": result["refined_query"],
                "decomposed_query": result["decomposed_query"],
                "retrieved_documents": result["retrieved_documents"]
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Query System with Decomposition')
    parser.add_argument('--query', type=str, help='Query to process')
    args = parser.parse_args()

    if args.query:
        try:
            result = process_query(args.query)
            print("\nQuery Results:")
            print(f"\nOriginal Query: {result['original_query']}")
            print(f"Refined Query: {result['refined_query']}")
            print(f"\nDecomposed Query:")
            print(json.dumps(result['decomposed_query'], indent=2))
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
        app.run(host="0.0.0.0", port=port, debug=True)
