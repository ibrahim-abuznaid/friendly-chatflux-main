from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def load_document(file_path):
    """Load the content of the markdown document"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading document: {e}")
        return None

def get_answer(question, context):
    """Get answer from GPT-4 based on the question and document context"""
    try:
        # Create system message with context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on the provided document. "
                    "Only answer based on the information in the document. "
                    "If the answer cannot be found in the document, say 'I cannot find this information in the document.'"
                )
            },
            {
                "role": "user", 
                "content": f"Here is the document content:\n\n{context}\n\nQuestion: {question}"
            }
        ]

        # Call GPT-4
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4 Turbo
            messages=messages,
            temperature=0,  # Lower temperature for more focused answers
            max_tokens=1000  # Adjust based on expected answer length
        )


        return response.choices[0].message.content

    except Exception as e:
        return f"Error getting answer: {e}"

def main():
    # Load the document
    doc_path = "docx.md"  # Update path as needed
    document_content = load_document(doc_path)
    
    if not document_content:
        print("Failed to load document")
        return

    # Main interaction loop
    print("Ask questions about the document (type 'quit' to exit)")
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        if question:
            answer = get_answer(question, document_content)
            print("\nAnswer:", answer)

if __name__ == "__main__":
    main() 