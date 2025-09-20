import os
import sys
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pinecone import Pinecone

load_dotenv()


# Add root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import logger

def get_qa_pipeline(index_name="rag-app-768", model_name="all-mpnet-base-v2", namespace="default"):
    """
    RetrievalQA pipeline using Groq LLM + Pinecone retriever.
    """
    print(f"QA pipeline initialized with Groq LLM + Pinecone retriever (namespace={namespace})")

    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 2. Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # 3. Load vectorstore with correct namespace
    vectorstore = PineconeStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,  # ✅ FIXED
    )

    # 4. Groq LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    # 5. Combine retriever + LLM
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    return qa

