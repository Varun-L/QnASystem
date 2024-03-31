from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def main():
    load_dotenv()
    st.set_page_config(page_title="Create your KB")
    st.header("Save Your KB")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=150,
        chunk_overlap=30,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      mn="hkunlp/instructor-large" #"hkunlp/instructor-xl"
      embeddings = HuggingFaceInstructEmbeddings(model_name=mn,model_kwargs={'device': 'cpu'})
      
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      knowledge_base.save_local("kb")
      st.write("Successfully saved Knowledge Base")
      
if __name__ == '__main__':
    main()
