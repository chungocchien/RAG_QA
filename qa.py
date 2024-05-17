
from langchain.chains import LLMChain, RetrievalQA
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import CTransformers, Ollama, LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

import streamlit as st

def read_vectordb():
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0",
                                  cohere_api_key='jgfGLUxBFOGVtZzQyNLcorveyREv7i8ENFRTQADk')
    db = FAISS.load_local(vector_db_path, embeddings)
    return db
def close_docs(question, db):
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k":3})
    return retriever.invoke(question)

def create_prompt(template):
    prompt= PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt
def load_file(model_file):
    llm = LlamaCpp( model_path=model_file,
                    n_ctx=2048,
                    temperature=0,
                    max_tokens=512,
                    verbose=True,
    )
    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

db = read_vectordb()
llm = load_file(model_file)

template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi.
Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt(template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

##initialize our streamlit app

st.set_page_config(page_title="Q&A Demo")

st.header("Gemini LLM Application")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

question = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and question:
    docs = close_docs(question, db)
    context = format_docs(docs)
    print(context)
    response = llm_chain.run({"context":context,"question": question})
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", question))
    st.subheader("The Response is")
    # for chunk in response:
    st.write(response)
    st.session_state['chat_history'].append(("Bot", response))
st.subheader("The Chat History is")

for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")

