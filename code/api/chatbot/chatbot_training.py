# chatbot_training.py
from pymongo import MongoClient
import gridfs
import io
import tempfile
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
import os
from pinecone import Pinecone, ServerlessSpec

def train_chatbot():
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Specifying index name and embedding model
    index_name = "heartdisease"
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Deletes current index so we get a fresh index with the new data 
    pc.delete_index(index_name)

    # Create a new index
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    # Loading data from MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["heart_disease"]
    fs = gridfs.GridFS(db)

    files = fs.find()
    documents = []

    def load_file_content(file, extension):
        if extension == "csv":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = CSVLoader(temp_file_path)
            content = loader.load()
            temp_file.close()
        elif extension == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            content = loader.load()
            temp_file.close()
        elif extension == "txt":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = TextLoader(temp_file_path)
            content = loader.load()
            temp_file.close()
        elif extension in ["doc", "docx"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = Docx2txtLoader(temp_file_path)
            content = loader.load()
            temp_file.close()
        elif extension in ["xlsx", "xls"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = UnstructuredExcelLoader(temp_file_path)
            content = loader.load()
            temp_file.close()
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        return content

    for file in files:
        grid_out = fs.get(file._id)
        file_content = grid_out.read()
        extension = file.filename.split('.')[-1].lower()

        try:
            loaded_content = load_file_content(io.BytesIO(file_content), extension)
            for content in loaded_content:
                if isinstance(content, Document):
                    doc = Document(page_content=content.page_content, metadata={"filename": file.filename})
                else:
                    doc = Document(page_content=content['text'], metadata={"filename": file.filename})
                documents.append(doc)
        except ValueError as e:
            print(e)

    # Splitting the documents
    r_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
    )

    split_documents = r_text_splitter.split_documents(documents)

    # Store the embeddings in Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        split_documents,
        index_name=index_name,
        embedding=embeddings,
    )

    print("Files embedded and stored in Pinecone successfully.")
