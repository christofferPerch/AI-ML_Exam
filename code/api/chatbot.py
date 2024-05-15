from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

index_name = "heartdisease"
embeddings = OpenAIEmbeddings()

# pdf_path = "heart_disease2.pdf"
# data = PyPDFLoader(pdf_path).load()

# r_text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=0,
#     separators=["\n\n", "\n", " ", ""],
# )
# split = r_text_splitter.split_documents(data)

# vectorstore = PineconeVectorStore.from_documents(
#         split,
#         index_name=index_name,
#         embedding=embeddings,
#     )

heart_disease_keywords = [
    "cardiovascular",
    "heart",
    "artery",
    "hypertension",
    "cholesterol",
    "myocardial",
    "angina",
]


def is_question_relevant(question):
    """Check if the question contains any of the keywords related to heart disease."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in heart_disease_keywords)


def chat(input):
    if not is_question_relevant(input):
        return "I don't have information on that topic. Thanks for asking, master!"

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        openai_api_key=openai_api_key, model_name="gpt-4-turbo", temperature=0.0
    )

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer!!!.
    Use five sentences maximum. Keep the answer as concise as possible.
    Always say "thanks for asking, master!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    question = "Who wrote Prevalence of Uncontrolled Risk Factors for Cardiovascular Disease: United States, 1999-2010"

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(input)


response = chat("What is a heart")
print(response)
