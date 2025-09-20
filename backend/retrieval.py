import os
from dotenv import load_dotenv
import sys
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore

from langchain_community.embeddings import HuggingFaceEmbeddings


# Add root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger
# Load env variables
load_dotenv()


def retrieve_from_pinecone(
    query: str,
    index_name: str = "rag-app-768",
    model_name: str = "all-mpnet-base-v2",
    top_k: int = 3
):
    """
    Retrieve top-k most relevant chunks from Pinecone given a query.
    """
    try:
        # 1. Load embedding model
        # 1. Create embedding model
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        embedding_vector = embeddings.embed_query("test")
        embedding_dim = len(embedding_vector)

        # 2. Connect to Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logger.info(f"Connected to Pinecone index: {index_name}")

        # 3. Create VectorStore object
        vectorstore = PineconeStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        # 4. Perform similarity search
        results = vectorstore.similarity_search(query, k=top_k)
        logger.info(f"Retrieved {len(results)} chunks for query: {query}")

        return results

    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        return []

print(retrieve_from_pinecone("What courses are in 5th semester?"))
