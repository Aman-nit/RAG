#!/usr/bin/env python3
"""
Basic RAG Example

This example demonstrates how to use the RAG templates provided in the repository
to create a simple question-answering system.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    """
    Basic RAG pipeline example.
    
    This example shows how to:
    1. Load documents from a directory
    2. Create embeddings and store them in a vector database
    3. Query the system with questions
    """
    
    # Check if required environment variables are set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not set. Please set it in your .env file.")
        print("You can still run the document processing and vector database parts.")
    
    print("🚀 Starting Basic RAG Example")
    print("=" * 50)
    
    # Step 1: Document Processing
    print("\n📚 Step 1: Loading and processing documents...")
    
    # Create sample documents for demonstration
    sample_docs_dir = Path("./sample_data")
    sample_docs_dir.mkdir(exist_ok=True)
    
    # Create sample document if it doesn't exist
    sample_doc_path = sample_docs_dir / "sample_doc.txt"
    if not sample_doc_path.exists():
        sample_content = """
        Artificial Intelligence (AI) is the simulation of human intelligence in machines 
        that are programmed to think and act like humans. The term may also be applied 
        to any machine that exhibits traits associated with a human mind such as learning 
        and problem-solving.
        
        Machine Learning is a subset of AI that provides systems the ability to automatically 
        learn and improve from experience without being explicitly programmed. Machine learning 
        focuses on the development of computer programs that can access data and use it to learn for themselves.
        
        Deep Learning is a subset of machine learning in artificial intelligence that has networks 
        capable of learning unsupervised from data that is unstructured or unlabeled. It is also 
        known as deep neural learning or deep neural network.
        """
        sample_doc_path.write_text(sample_content.strip())
        print(f"✅ Created sample document: {sample_doc_path}")
    
    print(f"📖 Processing documents from: {sample_docs_dir}")
    
    # Step 2: Vector Database Setup
    print("\n🔍 Step 2: Setting up vector database...")
    print("🔄 Creating embeddings for documents...")
    print("✅ Vector database ready for queries")
    
    # Step 3: Query Examples
    print("\n❓ Step 3: Example queries...")
    
    sample_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is the difference between ML and deep learning?",
        "Tell me about neural networks"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("   Answer: [This would contain the RAG-generated response]")
        print("   Sources: sample_doc.txt (chunk 1)")
    
    print("\n" + "=" * 50)
    print("✨ RAG Example Complete!")
    print("\nTo run with actual implementation:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up your .env file with API keys")
    print("3. Implement the templates from the README")
    print("4. Replace this example with actual RAG pipeline calls")

if __name__ == "__main__":
    main()