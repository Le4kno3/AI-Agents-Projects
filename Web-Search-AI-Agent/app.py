from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

class EnchancedWebScraperTool:
    def __init__(self):
        self.name = "WebScrapper"
        self.description = "Scrape content from a website with advanced options. Input should be URL"

    def run(self, url: str):
        try:
            # Send the request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)

            # Raise an error for bad responses
            response.raise_for_status()

            # Parse json response.
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()
        
            # Remove unwanted response elements
            for script in soup(["script", "style","footer","nav","aside"]):
                script.extract()  # remove all javascript and stylesheet code

            text = soup.get_text(separator="\n", strip=True)

            lines = (line.strip() for line in text.splitlines())

            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

            text = "\n".join(chunk for chunk in chunks if chunk)

            max_length = 4000

            if len(text) > max_length:
                text = text[:max_length] + "...\n[Content truncated due to length]"

            return f"Content from {url}:\n\n{text}"
        except Exception as e:
            return f"Error occurred while scraping {url}:\n\n{str(e)}"

st.set_page_config(
        page_title="Web Search AI Agent",
        page_icon=":mag_right:",
        layout="wide",
        initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("Web Search AI Agent")
    st.write("This tool allows you to ask questions and get answers from the web.")

    st.subheader("How it works")
    st.write('''
    1. Enter your question in the input field.
    2. Click the "Search" button.
    3. The AI agent will search the web and provide an answer.
        - Search web for relevant information
        - Extract content from websites
        - Generate a comprehensive answer.
    ''')

    st.subheader("Settings")

    model_choice = st.selectbox("Choose a model:", ["GPT-4o", "GPT-4o-mini"], index=0)

    temperature = st.slider(
        "Select temperature (creativity):", min_value=0.0, max_value=1.0, value=0.5, step=0.1
    )

    max_iterations = st.slider(
        "Max search iterations:", min_value=1, max_value=10, value=5, step=1
    )

st.title("Web Search AI Agent")
st.write("Ask me anything, and I'wll search the web for you!")

@st.cache_resource
def initialize_web_agent(model_name, temp, max_iter):
    api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temp,
        max_iterations=max_iter,
        openai_api_key=api_key
    )

    agent_type = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION

    search_tool = DuckDuckGoSearchRun()

    web_scraper_tool = EnchancedWebScraperTool()

    tools = [
        Tool(name='Search', func=search_tool.run, description="Use this tool to search the web for relevant information. Input should be a search query."),
        Tool(name='WebScraper', func=web_scraper_tool.run, description="Use this tool to scrape content from a website. Input should be a URL.")
    ]

    agent = initialize_agent(
        tools = tools, 
        llm = llm, 
        agent=agent_type, 
        verbose=True,
        max_iterations=max_iter,
        handle_parsing_errors = True,
        early_stopping_method = "generate"
    )

    return agent

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("What would you like to know")

try:
    agent = initialize_web_agent(
        model_choice, 
        temperature, 
        max_iterations
    )
except Exception as e:
    st.error(f"Error initializing web agent: {e}")
    agent = None

if user_question and agent:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.status("Working on it...", expanded=True):
            st.write("Searching the web for an answer...")
            try:
                response = agent.run(user_question)
                status.update(label="Search Completed!", state="complete")
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                status.update(label="Search Failed", state="error")
                error_message = f"Error occurred while processing your request: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.error(error_message)