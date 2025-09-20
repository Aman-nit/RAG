# RAG - Retrieval-Augmented Generation

A comprehensive toolkit for building Retrieval-Augmented Generation (RAG) applications that combine the power of large language models with external knowledge sources.

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by providing them with relevant context from external knowledge sources. Instead of relying solely on the model's training data, RAG systems:

1. **Retrieve** relevant documents or information from a knowledge base
2. **Augment** the user's query with this retrieved context
3. **Generate** more accurate and contextually relevant responses

This approach helps reduce hallucinations, provides up-to-date information, and allows for domain-specific knowledge integration.

## Features

- 🔍 **Document Retrieval**: Efficient similarity search across large document collections
- 📚 **Knowledge Base Integration**: Support for multiple data sources and formats
- 🤖 **LLM Integration**: Compatible with various language models (OpenAI, Hugging Face, etc.)
- ⚡ **Vector Database Support**: Integration with popular vector databases
- 🛠️ **Customizable Pipeline**: Modular architecture for easy customization
- 📊 **Evaluation Tools**: Built-in metrics for RAG system performance

## Installation

```bash
# Clone the repository
git clone https://github.com/Aman-nit/RAG.git
cd RAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic RAG Pipeline

```python
from rag import RAGPipeline, DocumentLoader, VectorStore

# Initialize the RAG pipeline
pipeline = RAGPipeline(
    model_name="gpt-3.5-turbo",
    embedding_model="text-embedding-ada-002"
)

# Load and process documents
loader = DocumentLoader()
documents = loader.load_directory("./data/")

# Create vector store
vector_store = VectorStore()
vector_store.add_documents(documents)

# Query the system
query = "What is machine learning?"
response = pipeline.query(query, vector_store)
print(response)
```

## Templates

### 1. Document Ingestion Template

```python
# document_ingestion.py
import os
from typing import List
from dataclasses import dataclass

@dataclass
class Document:
    content: str
    metadata: dict
    
class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, directory_path: str) -> List[Document]:
        """Load documents from a directory."""
        documents = []
        
        for filename in os.listdir(directory_path):
            if filename.endswith(('.txt', '.md', '.pdf')):
                file_path = os.path.join(directory_path, filename)
                content = self._read_file(file_path)
                
                # Create document chunks
                chunks = self._chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        content=chunk,
                        metadata={
                            'source': filename,
                            'chunk_id': i,
                            'file_path': file_path
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    def _read_file(self, file_path: str) -> str:
        """Read content from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks

# Usage example
if __name__ == "__main__":
    processor = DocumentProcessor()
    docs = processor.load_documents("./data/")
    print(f"Loaded {len(docs)} document chunks")
```

### 2. Vector Database Template

```python
# vector_database.py
import numpy as np
from typing import List, Tuple
import faiss
from sentence_transformers import SentenceTransformer

class VectorDatabase:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector database."""
        self.documents.extend(documents)
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        self.embeddings.extend(embeddings)
        
        # Build or update FAISS index
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index from embeddings."""
        if self.embeddings:
            embeddings_array = np.array(self.embeddings).astype('float32')
            dimension = embeddings_array.shape[1]
            
            self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
            self.index.add(embeddings_array)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if not self.index:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results

# Usage example
if __name__ == "__main__":
    # Initialize vector database
    vector_db = VectorDatabase()
    
    # Add documents (assuming you have documents from the previous template)
    processor = DocumentProcessor()
    documents = processor.load_documents("./data/")
    vector_db.add_documents(documents)
    
    # Search
    results = vector_db.search("machine learning algorithms", top_k=3)
    for doc, score in results:
        print(f"Score: {score:.3f}")
        print(f"Content: {doc.content[:200]}...")
        print("---")
```

### 3. RAG Query Pipeline Template

```python
# rag_pipeline.py
import openai
from typing import List, Optional
import logging

class RAGPipeline:
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
        max_context_length: int = 4000
    ):
        openai.api_key = openai_api_key
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.vector_db = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def set_vector_database(self, vector_db: VectorDatabase):
        """Set the vector database for retrieval."""
        self.vector_db = vector_db
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.1
    ) -> dict:
        """Process a query through the RAG pipeline."""
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(question, top_k)
            
            # Step 2: Prepare context
            context = self._prepare_context(retrieved_docs)
            
            # Step 3: Generate response
            response = self._generate_response(question, context, temperature)
            
            return {
                'answer': response,
                'sources': [doc.metadata for doc, _ in retrieved_docs],
                'context_used': context[:500] + "..." if len(context) > 500 else context
            }
            
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {str(e)}")
            return {'error': str(e)}
    
    def _retrieve_documents(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents from vector database."""
        if not self.vector_db:
            raise ValueError("Vector database not set. Use set_vector_database() first.")
        
        return self.vector_db.search(query, top_k)
    
    def _prepare_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        current_length = 0
        
        for doc, score in retrieved_docs:
            doc_text = f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.content}\n"
            
            if current_length + len(doc_text) > self.max_context_length:
                break
                
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n---\n".join(context_parts)
    
    def _generate_response(self, question: str, context: str, temperature: float) -> str:
        """Generate response using OpenAI API."""
        prompt = f"""
        Based on the following context, please answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

        Context:
        {context}

        Question: {question}

        Answer:
        """
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline(openai_api_key="your-api-key-here")
    
    # Setup vector database (from previous templates)
    vector_db = VectorDatabase()
    processor = DocumentProcessor()
    documents = processor.load_documents("./data/")
    vector_db.add_documents(documents)
    pipeline.set_vector_database(vector_db)
    
    # Query the system
    result = pipeline.query("What is machine learning?")
    print("Answer:", result['answer'])
    print("Sources:", result['sources'])
```

### 4. Configuration Template

```python
# config.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
from pathlib import Path

@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_sequence_length: int = 512

@dataclass
class VectorDBConfig:
    type: str = "faiss"  # faiss, pinecone, weaviate, etc.
    dimension: int = 384
    similarity_metric: str = "cosine"
    
    # Pinecone specific
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None

@dataclass
class LLMConfig:
    provider: str = "openai"  # openai, huggingface, anthropic
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 500
    
@dataclass
class DocumentConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.txt', '.md', '.pdf', '.docx']

@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = None
    vector_db: VectorDBConfig = None
    llm: LLMConfig = None
    document: DocumentConfig = None
    
    # RAG specific settings
    top_k_retrieval: int = 5
    max_context_length: int = 4000
    
    # Paths
    data_directory: str = "./data"
    cache_directory: str = "./cache"
    
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.vector_db is None:
            self.vector_db = VectorDBConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.document is None:
            self.document = DocumentConfig()
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # LLM configuration from environment
        config.llm.api_key = os.getenv('OPENAI_API_KEY')
        config.llm.model_name = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        
        # Vector DB configuration
        config.vector_db.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        config.vector_db.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
        config.vector_db.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
        
        # Paths
        config.data_directory = os.getenv('DATA_DIRECTORY', './data')
        config.cache_directory = os.getenv('CACHE_DIRECTORY', './cache')
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration."""
        if self.llm.provider == "openai" and not self.llm.api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.vector_db.type == "pinecone":
            if not all([
                self.vector_db.pinecone_api_key,
                self.vector_db.pinecone_environment,
                self.vector_db.pinecone_index_name
            ]):
                raise ValueError("Pinecone configuration is incomplete")
        
        return True

# Usage example
if __name__ == "__main__":
    # Create default configuration
    config = RAGConfig()
    
    # Or create from environment variables
    config = RAGConfig.from_env()
    
    # Validate configuration
    try:
        config.validate()
        print("Configuration is valid")
    except ValueError as e:
        print(f"Configuration error: {e}")
```

## Environment Setup

Create a `.env` file in your project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Other LLM providers
HUGGINGFACE_API_KEY=your_huggingface_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Vector Database Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name

# Paths
DATA_DIRECTORY=./data
CACHE_DIRECTORY=./cache

# Model Configuration
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Project Structure

```
RAG/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── document_ingestion.py
│   ├── vector_database.py
│   ├── rag_pipeline.py
│   ├── config.py
│   └── utils.py
├── data/
│   └── documents/
├── tests/
│   ├── test_ingestion.py
│   ├── test_vector_db.py
│   └── test_pipeline.py
└── examples/
    ├── basic_rag.py
    ├── custom_embeddings.py
    └── evaluation.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/

# Run type checking
mypy src/
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the open-source community for providing excellent tools and libraries
- Inspired by research in retrieval-augmented generation and information retrieval
- Built with ❤️ for the AI community

## Roadmap

- [ ] Support for more document formats (PDF, DOCX, etc.)
- [ ] Integration with more vector databases (Weaviate, Qdrant, etc.)
- [ ] Advanced retrieval strategies (hybrid search, re-ranking)
- [ ] Evaluation framework for RAG systems
- [ ] Web interface for easy interaction
- [ ] API endpoints for integration with other applications