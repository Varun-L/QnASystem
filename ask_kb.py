from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import numpy as np

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    mn="hkunlp/instructor-large" #"hkunlp/instructor-xl"
    embeddings = HuggingFaceInstructEmbeddings(model_name=mn,model_kwargs={'device': 'cpu'})
    knowledge_base = FAISS.load_local("kb",embeddings=embeddings)
    
    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        # llm = OpenAI()
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question) 
        st.write(response)

if __name__ == '__main__':
    main()
