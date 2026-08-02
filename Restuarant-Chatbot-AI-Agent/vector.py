from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document   # Create documents -> pass to chrome
import os
import pandas as pd

DB_PATH = "./database/chrome_langchain_db"

# Create the data frame
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Create embedding models
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# "add_document" is just a variable to check if vector db exists or not.
add_document = not os.path.exists(DB_PATH)

# Check if the vector database already exists, if not then it embed text to create vector documents.
if add_document:
    documents = []
    ids = []

    for index, row in df.iterrows():
        # Convert this text row in a --> Document (of vector database)
        document = Document(
            page_content=row['Title']+ " " + row["Review"],  # the data that we want to convert to document.
            metadata = {        # we capute this data, but we dont use it in vector database searching
                "rating": row["Rating"],
                "date": row["Date"]
            },
            id = str(index)    # unique id for each document
        )
        ids.append(str(index))
        documents.append(document)

# Create the vector store using chroma
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=DB_PATH,
    embedding_function=embedding_model
)

# Again check if vector database does not exists, now store the documents in the vector database.
# This will embedded documents in the vector store.
if add_document:
    vector_store.add_documents(documents=documents, ids=ids)

# Lookup documents
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # number of relevant documents (reviews) to retrieve
)