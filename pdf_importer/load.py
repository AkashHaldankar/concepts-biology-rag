import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings

# Load env variables
load_dotenv()

# A Directory loader to get pdf in data folder
loader = DirectoryLoader(
    os.path.abspath("../data"),
    glob="**/*.pdf",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    sample_size=1,
    loader_cls=UnstructuredPDFLoader,
)

# Load docs
docs = loader.load()

# Generate Embeddings from the Open AI Model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Use Semantic Chunker to split text
text_splitter = SemanticChunker(
    embeddings=embeddings
)

# Crate chunks from text_splitter
chunks = text_splitter.split_documents(docs)

# Insert data into PGVector
PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=os.getenv("PG_COLLECTION"),
    connection_string=os.getenv("POSTGRES_URL"),
    pre_delete_collection=True,
)
