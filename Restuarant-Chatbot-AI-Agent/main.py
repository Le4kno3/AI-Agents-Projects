from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.1:8b")

# Query template
queryTemplate = """
You are an expert in answering questions about restaurants

Here are some relevant review: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(queryTemplate)

# chain the prompt and model
chain = prompt | model

while True:
    print("\n\n--------------------------------------------------------\n")
    user_input = input("Enter your question (or type 'exit' to quit): ")
    print("\n--------------------------------------------------------\n")
    # while loop break condition
    if user_input.lower() in ['exit', 'quit', 'q']:
        break

    # Lookup for relevant documents (reviews) based on the user input
    # How many reviews to retrieve, will depend on the "search_kwargs".
    # retriever will automatically embed the user input.
    reviews = retriever.invoke(user_input)

    # run the model, and pass only relevant reviews, based on the user.
    result = chain.invoke({
        "reviews": reviews,
        "question": user_input
    })

    # print the results
    print(result)