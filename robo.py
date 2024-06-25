from utils.rag import *
import streamlit as st
from dataclasses import dataclass
from llama_index.llms.ollama import Ollama


@st.cache_resource(show_spinner=False)
def preparebot():
    collection=Ingestion(path='data/SPOTIFY_REVIEWS_SMALL.csv',embedding_model="BAAI/bge-small-en-v1.5",rowlimit=2000)
    collection.run()
    bot=RAGBot(collection=collection,MODEL_NAME="gemma:2b",REQUEST_TIMEOUT=60)
    return bot

bot=preparebot()

@dataclass
class Message:
    actor: str
    payload: str


USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Holaa, I can help answer your question based Spotify app reviews.")]

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    # response: str = llm.complete(prompt).text
    response: str = bot.qna(question=prompt,RAG_KEYWORD=60)
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response) 