from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re

def clean_text(text):
    """
    Clean text by normalizing whitespace and removing common headers/footers
    """
    # Normalize whitespace - replace multiple spaces, tabs, newlines with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common header/footer patterns
    # Remove page numbers (standalone numbers)
    text = re.sub(r'\b\d{1,4}\b(?=\s|$)', '', text)
    
    # Remove common footer patterns (copyright, website URLs, etc.)
    text = re.sub(r'©.*?\d{4}', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Remove common header patterns (chapter numbers, section markers)
    text = re.sub(r'^(Chapter|Section)\s+\d+.*?$', '', text, flags=re.MULTILINE)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', '---', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def load_pdf_docs(path:str=None):
    # Load PDF files
    PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT_PATH, "data/raw_pdfs")
    # FILE_PATH = os.path.join(DATA_DIR, 'German Grammer Guide.pdf')
    FILE_PATH = os.path.join(DATA_DIR, 'RLbook2020_Sutton.pdf')

    # load the PDF file
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    return docs

def clean_and_chunk_docs(docs):
    """
    Returns chunks of documents after cleaning and Text-splitting
    Input: docs
    Return: chunks
    """
    # Clean the text in each document
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split documents into chunks
    chunks = text_splitter.split_documents(docs)
    return chunks

if __name__=='__main__':
    docs = load_pdf_docs()
    chunks = clean_and_chunk_docs(docs)
    print(f"Documents after cleaning: {len(docs)}")
    print(f"Chunks after splitting: {len(chunks)}")
    print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.1f} characters")

    # Display first few chunks as examples
    print("\n--- Sample chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} ({len(chunk.page_content)} chars):")
        print(f"'{chunk.page_content[:200]}...'")
        if hasattr(chunk, 'metadata'):
            print(f"Metadata: {chunk.metadata}")