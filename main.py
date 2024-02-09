import os
import dill
import streamlit as st
import pickle
import time
import langchain
import pprint
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import GooglePalmEmbeddings  # Updated
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_png_as_page_bg('background.png')


load_dotenv()  
google_palm_key = os.getenv('GOOGLE_API_KEY')

st.title("NAMI MY LOVE : News Research Tool 📈")
st.sidebar.title("News Article URLs")

# ... rest of your code


urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", "")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vectorindex_palm.pkl"

main_placeholder = st.empty()
llm = GooglePalm(google_api_key=google_palm_key, temperature=0.9, max_tokens=500)

if process_url_clicked:
    
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    embeddings = GooglePalmEmbeddings(google_api_key=google_palm_key)
    vectorindex_palm = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    with open(file_path, "wb") as f:
        dill.dump(vectorindex_palm, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorindex_palm = dill.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_palm.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
           
            st.header("Answer")
            st.write(result["answer"])

            
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  
                for source in sources_list:
                    st.write(source)
