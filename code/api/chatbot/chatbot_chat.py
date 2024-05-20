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

def chat(input):
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

    retriever=vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={
                              'score_threshold': 0.5})

    llm = ChatOpenAI(
        openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.0
    )

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer!!!.
    Use five sentences maximum. Keep the answer as concise as possible. 

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
