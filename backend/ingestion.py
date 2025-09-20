import os
import re
import sys
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)

# Add root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import logger

# Load environment variables
load_dotenv()


#Loading Files?folders in langchain documnets
def load_data(file_path: str) -> List[Document]:
    """Load data from different file types."""
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)
        elif ext in [".xls", ".xlsx"]:
            loader = UnstructuredExcelLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error while loading {file_path}: {e}")
        return []


#Splitting documents into smaller chunks
def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """Split documents into smaller chunks for better embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunks")
    return chunks

#cleaning document text
def clean_documents(documents):
    """Clean up document text."""
    cleaned_docs = []
    for doc in documents:
        text = doc.page_content
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        cleaned_docs.append(
            Document(page_content=text, metadata=doc.metadata)
        )
    return cleaned_docs


def store_in_pinecone(documents, index_name, namespace="default", model_name="all-mpnet-base-v2"):
    """
    Store document chunks in Pinecone with embeddings.
    Uses namespace to separate user data inside the same index.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        embedding_vector = embeddings.embed_query("test")
        embedding_dim = len(embedding_vector)

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        existing_indexes = pc.list_indexes().names()

        if index_name in existing_indexes:
            index_info = pc.describe_index(index_name)
            if index_info.dimension != embedding_dim:
                logger.warning(
                    f"Index dim mismatch. Expected {embedding_dim}, got {index_info.dimension}. Creating new index with correct dimension."
                )
                index_name = f"{index_name}-{embedding_dim}"
                if index_name not in existing_indexes:
                    pc.create_index(
                        name=index_name,
                        dimension=embedding_dim,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    logger.info(f"Created new Pinecone index: {index_name}")
        else:
            logger.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Created Pinecone index: {index_name}")

        logger.info(f"Storing {len(documents)} documents in Pinecone index '{index_name}', namespace '{namespace}'...")
        vectorstore = PineconeStore.from_documents(
            documents,
            embeddings,
            index_name=index_name,
            namespace=namespace
        )
        logger.info("File embedded successfully!")
        return vectorstore

    except Exception as e:
        logger.error(f"Error storing in Pinecone: {e}")
        return None




# Main Ingestion Function 
def process_pdf(file_path: str, namespace: str = "default"):
    """
    Full ingestion pipeline for a PDF.
    Loads -> splits -> cleans -> stores in Pinecone (inside a namespace).
    """
    index_name = "rag-app-768"  # Use a single index for all users
    logger.info(f"Starting ingestion for {file_path} into index {index_name}, namespace={namespace}")  

    # Load, split, clean
    docs = load_data(file_path)
    if not docs:
        raise ValueError(f"No documents loaded from {file_path}")

    chunks = split_documents(docs)
    clean_text = clean_documents(chunks)

    # Store in Pinecone with namespace
    vectorstore = store_in_pinecone(clean_text, index_name=index_name, namespace=namespace)

    logger.info("Ingestion process completed.")
    return {"chunks": len(clean_text), "stored": vectorstore is not None}


