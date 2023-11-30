from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

app = FastAPI()

class UserQuestion(BaseModel):
    user_id: str
    question: str

class Message(BaseModel):
    content: str

class ConversationResponse(BaseModel):
    chat_history: List[Message]
    api_calls: int

pdf_directory = "pdf"
load_dotenv()

# Initialize these variables outside the endpoint to persist state across requests
user_sessions = {}
api_calls_counter = {}

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def initialize_user(user_id):
    raw_text = ""
    for pdf_filename in os.listdir(pdf_directory):
        if pdf_filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_filename)
            with open(pdf_path, "rb") as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()

    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    return get_conversation_chain(vectorstore)

@app.post("/ask-question", response_model=ConversationResponse)
async def ask_question(user_question: UserQuestion):
    global user_sessions, api_calls_counter
    user_id = user_question.user_id
    if user_id not in user_sessions:
        user_sessions[user_id] = initialize_user(user_id)

    conversation_chain = user_sessions[user_id]
    response = conversation_chain({'question': user_question.question})
    chat_history = response['chat_history']

    # Update API calls counter
    api_calls_counter[user_id] = api_calls_counter.get(user_id, 0) + 1

    return {"chat_history": chat_history, "api_calls": api_calls_counter[user_id]}
