from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders.pdf import PyPDFLoader 
from langchain.document_loaders.directory import DirectoryLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import *


def faiss_vector_db():

    dir_loader = DirectoryLoader(
        DATA_DIR_PATH,
        glob= '*.pdf',
        loader_cls=PyPDFLoader
    )

    docs = dir_loader.load()
    print("PDFs Loaded")


    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap =CHUNK_OVERLAP 
    )

    inp_txt = txt_splitter.split_documents(docs)
    print("Data Chunk Created")

    hfembeddings = HuggingFaceHubEmbeddings(
        model=EMBEDDER,
       # model_kwargs={'device':'cpu'}
    )

    db = FAISS.from_documents(inp_txt, hfembeddings)
    db.save_local(VECTOR_DB_PATH)
    print("Vector store Creation Completed")

    

if __name__ == "__main__":
    faiss_vector_db()