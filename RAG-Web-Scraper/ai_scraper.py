import streamlit as st

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Model Name
MODEL_NAME = "llama3.1:8b"

# Copied from langchain platform
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Define Embedding model (text to vectors)
embeddings = OllamaEmbeddings(model=MODEL_NAME)
# Alternative to ChromaDB
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model=MODEL_NAME)

#######################################################################################################################
# Generate relevant Context (documents) for RAG Processing
def load_page(url):
    # Fetch the webpage contents (we dont need to clean it, as langchain handles it)
    loader = SeleniumURLLoader(
        urls=[url]
    )

    # Create document (vector embeddings) from the contents for use with langchain
    documents = loader.load()

    # return the list of documents
    return documents

# To split documents, so that we dont feed long documents to vector store, which inturn will be sent to LLM. LLM should not be given long documents for accuracy reasons.
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    data = text_splitter.split_documents(documents)
    return data # Returns the list of documents, splitted based on max document size.


#######################################################################################################################
# Add the documents to the vector store.
def index_docs(documents):
    vector_store.add_documents(documents)

# Retrieve documents from vector store, relevant to the user search query
def retrieve_docs(query):
    return vector_store.similarity_search(query)


#######################################################################################################################
# Feed the vector store response (context) and the user query to the LLM.
def answer_question(question, context):
    # Build prompt using template
    prompt = ChatPromptTemplate.from_template(template)
    # Combine prompt with model for processing
    chain = prompt | model
    # Invoke the chain with the question and context
    return chain.invoke({"question": question, "context": context})

#########################################################################################################################
# Design the UI & Orchestrate Processing
st.title("AI Crawler")
url = st.text_input("Enter URL:")

# Step 1: Load the webpage
documents = load_page(url)

# Step 2: Split the documents into smaller chunks
chunked_documents = split_text(documents)

# Step 3: Add the documents into the vector store
index_docs(chunked_documents)

# Step 4: Get user question from UI now.
question = st.chat_input()

# Step 5: Processing the question. Retrieve relevant documents and output in UI.
if question:
    # 5.1 Print the question on chat UI
    st.chat_message("user").write(question)
    # 5.2 Retrieve documents from vector store
    retrieve_documents = retrieve_docs(question)
    # 5.3 Combine the context for the LLM processing
    context = "\n\n".join([doc.page_content for doc in retrieve_documents])
    # 5.4 Feed the question and the context to the LLM & Get the answer from the LLM
    answer = answer_question(question, context)
    # 5.5 Print the answer on chat UI
    st.chat_message("assistant").write(answer)