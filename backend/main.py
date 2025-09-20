import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.pipeline import get_qa_pipeline
from fastapi import HTTPException,Form , Query
from pydantic import BaseModel
import uuid
from pinecone import pinecone
from dotenv import load_dotenv
from backend.ingestion import process_pdf  # your PDF ingestion function

app = FastAPI()
load_dotenv()

# CORS setup (so frontend can talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/upload")
async def upload_file(file: UploadFile = File(...), namespace: str = Form("default")):
    """
    Uploads and processes a file, storing embeddings in Pinecone under a user-specific namespace.
    """
    try:
        os.makedirs("data", exist_ok=True)  # store in data folder
        file_path = f"data/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process file (embed + store in Pinecone with namespace)
        process_pdf(file_path, namespace=namespace)

        return JSONResponse(content={"message": f"File uploaded and processed successfully in namespace '{namespace}'!"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/ask")
async def ask_question(payload: dict):
    """
    Accepts a question and optional namespace to search in user-specific data.
    """
    # Initialize QA pipeline
    qa_pipeline = get_qa_pipeline()
    try:
        query = payload.get("question")
        namespace = payload.get("namespace", "default")

        if not query:
            return JSONResponse(content={"error": "Question cannot be empty"}, status_code=400)

        # Pass namespace to QA pipeline
        result = qa_pipeline.invoke({"query": query, "namespace": namespace})

        sources = []
        if "source_documents" in result:
            sources = [doc.metadata for doc in result["source_documents"]]

        return JSONResponse(content={
            "answer": result.get("result", "No answer found."),
            "sources": sources,
            "namespace": namespace
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
from pinecone import Pinecone

import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  
index = pc.Index("rag-app-768")
@app.delete("/delete-namespace")
async def delete_user_namespace(namespace: str = Query(...,description='The namespace to delete.')):
    try:
        index = pc.Index("rag-app-768")
        index.delete(delete_all=True, namespace=namespace)
        return {"message": f"Namespace '{namespace}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
