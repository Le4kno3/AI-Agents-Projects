from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

## Fetch the CSV file, and store each line in an array
with open("realistic_restaurant_reviews.csv", "r") as file:
    reviews = file.readlines()

model = OllamaLLM(model="llama3.1:8b")

# Query template
queryTemplate = """
You are an expert in answering questions about restaurants

Here are some relevant review: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(queryTemplate)

chain = prompt | model

while True:
    print("\n\n--------------------------------------------------------")
    user_input = input("Enter your question (or type 'exit' to quit): ")
    print("\n\n")
    # while loop break condition
    if user_input.lower() in ['exit', 'quit', 'q']:
        break
    # setting the inputs to query
    result = chain.invoke({
        "reviews": reviews,     # without using the power of vector database / vector search, hence speed is slow
        "question": user_input
    })
    print(result)