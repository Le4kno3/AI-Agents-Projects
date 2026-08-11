from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# Model Name
MODEL_NAME = "llama3.1:8b"

# Prepare LLM Model
model = OllamaLLM(model=MODEL_NAME)

# Prepare and connect with firecrawl MCP server
server_params = StdioServerParameters(
    command="npx",
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    },
    args=["firecrawl-mcp"]
)


async def main():
    # Connecting the AI Agent with the MCP server
    async with stdio_client(server_params) as (read, write):
        # Creating a new client session
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Load MCP tools from our session
            tools = await load_mcp_tools(session)

            # Create the AI Agent using model & Firecrawl tools
            agent = create_agent(model, tools)

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to help the user."
                }
            ]

            # Step 1: Print all available tools in firecrawl
            print("Available Tools -", *[tool.name for tool in tools])
            print("-" * 60)

            # Step 2: In a menu format, take user input and process it.
            while True:
                user_input = input("\nYou: ")
                if user_input == "quit":
                    print("Goodbye")
                    break

                messages.append({"role": "user", "content": user_input[:175000]})   # Trim user input, incase user gives a very long message

                # Call the Firecrawl tool
                try:
                    agent_response = await agent.ainvoke({"messages": messages})

                    ai_message = agent_response["messages"][-1].content # To get the last message
                    print("\nAgent:", ai_message)
                except Exception as e:
                    print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())