import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import openai
from langchain_core.documents import Document
import re  # Add this import at the top with other imports
import argparse  # Add this import at the top

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

# Query Translation Step
@traceable(name="query_translation")
def translate_query(query):
    """Translate or refine the query with LangSmith tracing"""
    translation_prompt = ChatPromptTemplate.from_template(
        "Rewrite the following query to make it more clear and specific for a search: {query}"
    )
    translation_chain = translation_prompt | llm
    translated_query = translation_chain.invoke({"query": query}).content
    return translated_query

# Document Preparation
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

# Vector Store Indexing
@traceable(name="vector_store_creation")
def create_vector_store(document_splits):
    """Create a vector store from document splits with tracing"""
    vectorstore = Chroma.from_documents(
        documents=document_splits, 
        embedding=embeddings
    )
    return vectorstore

# Traceable Retrieval
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

# Retrieval and RAG Chain
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

# Main RAG Pipeline
@traceable(name="rag_pipeline")
def process_query(user_query):
    """Complete RAG pipeline with comprehensive LangSmith tracing"""
    try:
        # Load documents
        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(script_dir, "docx.md")
        loader = TextLoader(docs_path, encoding="utf-8")
        documents = loader.load()

        # Query Translation
        refined_query = translate_query(user_query)
        
        # Document Preparation
        document_splits = prepare_documents(documents)
        
        # Vector Store Creation
        vectorstore = create_vector_store(document_splits)
        
        # Document Retrieval
        retrieved_docs = retrieve_documents(vectorstore, refined_query)
        
        # RAG Chain
        rag_chain = create_rag_chain(vectorstore)
        
        # Generate Response
        response = rag_chain.invoke(refined_query)
        
        return {
            "original_query": user_query,
            "refined_query": refined_query,
            "response": response.content,
            "retrieved_documents": retrieved_docs
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
                "retrieved_documents": result["retrieved_documents"]
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='RAG Query System')
    parser.add_argument('--query', type=str, help='Query to process')
    args = parser.parse_args()

    if args.query:
        # Direct query processing
        try:
            result = process_query(args.query)
            print("\nQuery Results:")
            print(f"\nOriginal Query: {result['original_query']}")
            print(f"Refined Query: {result['refined_query']}")
            print(f"\nResponse: {result['response']}")
            print("\nRetrieved Documents:")
            for i, doc in enumerate(result['retrieved_documents'], 1):
                print(f"\nDocument {i}:")
                print(f"Page Number: {doc['metadata'].get('page_number', 'N/A')}")
                print(f"Content: {doc['page_content'][:200]}...")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    else:
        # Run as web server if no query provided
        port = int(os.getenv("PORT", 3001))
        app.run(host="0.0.0.0", port=port, debug=True) 