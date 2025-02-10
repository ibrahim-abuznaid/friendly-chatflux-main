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
import re
import argparse
from typing import List
import numpy as np

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

# RAG Fusion Query Generation
@traceable(name="generate_queries")
def generate_queries(question: str, n: int = 3) -> List[str]:
    """Generate multiple versions of the input question"""
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
    
    # Split the response into individual questions
    queries = [q.strip() for q in response.content.split('\n') if q.strip()]
    return queries[:n]

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
            # RRF formula: 1 / (k + rank)
            doc_scores[doc_id] += 1 / (k + rank + 1)
    
    # Create a mapping of doc_id to actual document
    doc_map = {}
    for doc_list in doc_lists:
        for doc in doc_list:
            doc_id = f"{doc.metadata.get('document_id')}_{doc.metadata.get('page_number')}"
            doc_map[doc_id] = doc
    
    # Sort documents by their fusion scores
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    # Return the documents in their fused order
    return [doc_map[doc_id] for doc_id in sorted_doc_ids]

# Document Preparation
@traceable(name="document_preparation")
def prepare_documents(documents):
    """Split documents into chunks with tracing and add metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400
    )
    
    processed_docs = []
    last_seen_page_number = None
    
    for doc_idx, doc in enumerate(documents):
        splits = text_splitter.split_documents([doc])
        
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
    
    return processed_docs

# Vector Store Indexing
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

# Traceable Retrieval with RAG Fusion
@traceable(name="document_retrieval", run_type="retriever")
def retrieve_documents(vectorstore, query):
    """Retrieve documents using RAG Fusion"""
    # Generate multiple queries
    queries = generate_queries(query)
    
    # Get retrievals for each query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    doc_lists = [retriever.invoke(q) for q in queries]
    
    # Combine results using reciprocal rank fusion
    fused_docs = reciprocal_rank_fusion(doc_lists)
    
    # Format documents for return
    formatted_docs = [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata)
        }
        for doc in fused_docs[:4]  # Return top 4 after fusion
    ]
    
    return formatted_docs

# Create RAG Chain
@traceable(name="rag_chain_creation")
def create_rag_chain(vectorstore):
    """Create the RAG chain with the retriever"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    rag_prompt = ChatPromptTemplate.from_template(
        """Answer the following question based on the provided context. 
        If you cannot answer the question based on the context, say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
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
    """Complete RAG pipeline with RAG Fusion"""
    try:
        # Load documents
        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(script_dir, "docx.md")
        loader = TextLoader(docs_path, encoding="utf-8")
        documents = loader.load()

        # Document Preparation
        document_splits = prepare_documents(documents)
        
        # Vector Store Creation
        vectorstore = create_vector_store(document_splits)
        
        # Document Retrieval with RAG Fusion
        retrieved_docs = retrieve_documents(vectorstore, user_query)
        
        # Generate multiple queries for context
        generated_queries = generate_queries(user_query)
        
        # RAG Chain
        rag_chain = create_rag_chain(vectorstore)
        
        # Generate Response
        response = rag_chain.invoke(user_query)
        
        return {
            "original_query": user_query,
            "generated_queries": generated_queries,
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
                "generated_queries": result["generated_queries"],
                "retrieved_documents": result["retrieved_documents"]
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

def save_vectorstore(vectorstore, path="faiss_index"):
    """Save the FAISS index to disk"""
    vectorstore.save_local(path)

def load_vectorstore(path="faiss_index"):
    """Load the FAISS index from disk"""
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Fusion Query System')
    parser.add_argument('--query', type=str, help='Query to process')
    args = parser.parse_args()

    if args.query:
        # Direct query processing
        try:
            result = process_query(args.query)
            print("\nQuery Results:")
            print(f"\nOriginal Query: {result['original_query']}")
            print(f"Generated Queries: {result['generated_queries']}")
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
