import os
import pandas as pd
from sqlalchemy import create_engine

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore


load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

index_name = "heartdisease"
embeddings = OpenAIEmbeddings()


def load_settings():
    server_name = "localhost"
    database_name = "HeartDisease"
    engine = create_engine(
        f"mssql+pyodbc://@{server_name}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes"
    )

    # Define the query to select all columns except Id and CreatedAt
    query = """
        SELECT CB.[BasePrompt]
        ,CB.[Temperature]
        ,GM.[ModelName]
        FROM [HeartDisease].[dbo].[ChatBotSettings] CB
        INNER JOIN [HeartDisease].[dbo].[GPTModel] GM
        ON CB.[GPTModelId] = GM.[Id]
        WHERE CB.[Id] = 1 
    """

    # Load data into a pandas DataFrame
    df = pd.read_sql(query, engine)

    base_prompt = df["BasePrompt"].values[0]
    temperature = df["Temperature"].values[0]
    model_name = df["ModelName"].values[0]
    model_name = model_name.replace("\t", "")

    return base_prompt, temperature, model_name


base_prompt, temperature, model_name = load_settings()


def chat(input):
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key, model_name=model_name, temperature=temperature
    )

    template = (
        base_prompt
        + """

    {context}

    Question: {question}

    Helpful Answer:
    """
    )

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Format documents to string.
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(input)
