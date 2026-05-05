import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_pdfs(pdf_paths: list[str]) -> list:
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    return documents
def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)

def ingest_pdfs(pdf_paths: list[str]) -> list:
    docs = load_pdfs(pdf_paths)
    chunks = chunk_documents(docs)
    return chunks