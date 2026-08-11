from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END


import os
import json
from typing import TypedDict

from dotenv import load_dotenv
from imap_tools import MailBox, AND

load_dotenv()

IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_PORT = os.getenv("IMAP_PORT")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_FOLDER = 'INBOX'

LLM_MODEL = 'qwen3:8b'
CHAT_MODEL = 'qwen3:8b'

class ChatState(TypedDict):
    messages: list

def connect():
    mail_box = MailBox(IMAP_HOST, IMAP_PORT)
    mail_box.login(IMAP_USER, IMAP_PASSWORD, initial_folder=IMAP_FOLDER)
    return mail_box

@tool
def list_unread_emails():
    """
    Return a bullet list of all UNREAD message's UID, subject, date and sender.
    """
    print("List Unread Emails Tool Called")

    with connect() as mb:
        unread = list(mb.fetch(criteria = AND(seen=False), headers_only=True, mark_seen=False))
    
    if not unread:
        return "No unread emails found."

    response = json.dumps([
        {
            "uid": msg.uid,
            "subject": msg.subject,
            "date": msg.date.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "sender": msg.from_ # from_ instead of from, because from is a python keyword
        } for msg in unread
    ])

    return response
    
@tool
def summarize_email(uid):
    """
    Summarize an email given it's IMAP UID. Return a short summary of the e-mails content / body in plain text.
    """
    print('Summarize E-Mail Tool Called on',uid)
    
    with connect() as mb:
        mail = next(mb.fetch(AND(uid=uid), mark_seen=False), None)
        if not mail:
            return f"Could not summarize e-mail with UID {uid}."
        
        prompt = (
            "Summarize the email concisely:\n\n"
            f"Subject: {mail.subject}\n"
            f"From: {mail.from_}\n"
            f"Date: {mail.date}\n\n"
            f"{mail.text or mail.html}\n"
        )

        # TODO: Feed this into an LLM and return result
        return raw_llm.invoke(prompt).content

# LLM1 - Initialize LLM
llm = init_chat_model(LLM_MODEL, model_provider='ollama')
# Bind tools to this LLM1
llm = llm.bind_tools([list_unread_emails, summarize_email])

# LLM2 - Inferior LLM for doing basic summarization
raw_llm = init_chat_model(LLM_MODEL, model_provider='ollama')

###############################################################################################################
# Create Langgraph

def llm_node(state):
    response = llm.invoke(state['messages'])
    return {
        'messages': state['messages'] + [response]
    }

def router(state):
    last_message = state['messages'][-1]
    return 'tools' if getattr(last_message, 'tool_calls', None) else 'end'

tool_node = ToolNode([list_unread_emails, summarize_email])

def tools_node(state):
    result = tool_node.invoke(state)
    return {'messages': state['messages'] + result['messages']}

builder = StateGraph(ChatState)
builder.add_node('llm', llm_node)
builder.add_node('tools', tools_node)
builder.add_edge(START, 'llm')
builder.add_edge('tools', 'llm')
builder.add_conditional_edges('llm', router, {'tools': 'tools', 'end': END})

graph = builder.compile()

if __name__ == "__main__":
    state = {
        'messages': []
    }

    print('Type an instruction or "quit" to exit:\n')

    while True:
        user_input = input('> ')
        if user_input.lower() == 'quit':
            break

        state['messages'].append({'role': 'user', 'content': user_input})

        state = graph.invoke(state)
        
        print(state['messages'][-1].content, '\n')