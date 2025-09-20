import gradio as gr
from pipeline import get_qa_pipeline

qa_pipeline = get_qa_pipeline()

def chat(query, history):
    answer = qa_pipeline({"query": query})
    return answer["result"]

with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(fn=chat, title="RAG Chatbot")

demo.launch()
