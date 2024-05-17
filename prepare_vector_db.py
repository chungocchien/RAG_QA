import os
import getpass

import PyPDF2
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from reportlab.pdfgen import canvas
from pypdf import PdfReader
import google.generativeai as genai
from langchain_cohere import CohereEmbeddings
import pandas as pd

vector_db_path = "vectorstores/db_faiss"

def create_db_from_data():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader('data', glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                       cohere_api_key='jgfGLUxBFOGVtZzQyNLcorveyREv7i8ENFRTQADk')
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db
def create_db_from_files():
    f = open("chunk_luat_lao_dong.pdf", "r", encoding='utf-8')
    data = f.read()
    chunks = data.split('<START>')

    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                       cohere_api_key='jgfGLUxBFOGVtZzQyNLcorveyREv7i8ENFRTQADk')
    embeddings = embedding_model.embed_documents(chunks)
    query_embeddings = create_query_db()
    title_embeddings = create_title_db()
    ctx_embeddings = np.array(embeddings) + np.array(query_embeddings) + 0.5*np.array(title_embeddings)
    text_embedding_pairs = zip(chunks, ctx_embeddings)
    db = FAISS.from_embeddings(text_embedding_pairs, embedding_model)
    db.save_local(vector_db_path)
    return db

def create_query_db():
    question_data = pd.read_csv('cauhoi_llm.csv')
    question1s = question_data['Question_1'].tolist()
    question2s = question_data['Question_2'].tolist()
    question3s = question_data['Question_3'].tolist()
    question4s = question_data['Question_4'].tolist()
    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                       cohere_api_key='jgfGLUxBFOGVtZzQyNLcorveyREv7i8ENFRTQADk')
    embeddings_question_1 = embedding_model.embed_documents(question1s)
    embeddings_question_2 = embedding_model.embed_documents(question2s)
    embeddings_question_3 = embedding_model.embed_documents(question3s)
    embeddings_question_4 = embedding_model.embed_documents(question4s)
    embeddings_questions = (0.25*np.array(embeddings_question_1)
                            + 0.25*np.array(embeddings_question_2)
                            + 0.25*np.array(embeddings_question_3)
                            + 0.25*np.array(embeddings_question_4))
    return embeddings_questions

def create_title_db():
    with open('title.txt', 'r', encoding='utf-8') as file:
        titles = file.readlines()
    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0",
                                       cohere_api_key='jgfGLUxBFOGVtZzQyNLcorveyREv7i8ENFRTQADk')
    title_embeddings = embedding_model.embed_documents(titles)
    return title_embeddings

create_db_from_files()