import os
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
prompt_template = """Đưa ra tiu đề cho đoạn văm bản sau, ngoài ra không đưa ra thông tin gì thêm 
"{text}"
Trả lời:"""
prompt = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(llm=model, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

f = open("chunk_luat_lao_dong.pdf", "r", encoding='utf-8')
data = f.read()
chunks = data.split('<START>')
titles = []
for i in range(len(chunks)):
    print(i)
    try:
        title = str(stuff_chain.run([Document(chunks[i])]))
    except:
        title = chunks[i]
    titles.append(title)

with open('title.txt', "w", encoding="utf-8") as f:
    # Ghi nội dung mảng vào file
    for string in titles:
        f.write(str(string)+'\n')