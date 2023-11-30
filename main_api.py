from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Your existing code goes here
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

app = FastAPI()

class UserQuestion(BaseModel):
    question: str

@app.post("/ask")
def ask_question(user_question: UserQuestion):
    # Your existing Streamlit logic goes here
    response = handle_userinput(user_question.question)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
