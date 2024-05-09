import streamlit as st
import os
import uuid
import openai
from pydantic import BaseModel
from typing import Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.runnable import RunnablePassthrough
from unstructured.partition.pdf import partition_pdf

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__98ac856199fcc4c3eea0a"
os.environ["OPENAI_API_KEY"] = "sk-2eS4pCzWR2YYG9oDLccyiyv"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Predefined PDF path
pdf_files = ["/content/lebo103-1-3.pdf"]

def process_pdfs(pdf_filenames):
    results = []
    for filename in pdf_filenames:
        raw_pdf_elements = partition_pdf(
            filename=filename,
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=0,
        )
        results.extend(raw_pdf_elements)
    return results

def categorize_elements(pdf_results):
    class Element(BaseModel):
        type: str
        text: Any

    categorized_elements = []
    for element in pdf_results:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))
    return categorized_elements

def summary_by_type(type_elements):
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = summarize_chain.batch([e.text for e in type_elements if e.type == 'text'], {"max_concurrency": 5})
    table_summaries = summarize_chain.batch([e.text for e in type_elements if e.type == 'table'], {"max_concurrency": 5})
    return text_summaries, table_summaries

def setup_retriever(texts, text_summaries):
    vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(text_summaries)]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))
    return retriever

pdf_results = process_pdfs(pdf_files)
elements = categorize_elements(pdf_results)
text_summaries, table_summaries = summary_by_type(elements)
retriever = setup_retriever([e.text for e in elements], text_summaries + table_summaries)

st.title('PDF Content-Based Question Answering')
user_question = st.text_input("Enter your question:")

if user_question:
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}.
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser())
    response = chain.invoke(user_question)
    st.write("Answer:", response)
