# robotrag

# Details
* LLM: ollama gemma 2b
* RAG_Embedding:BAAI/bge-small-en-v1.5
* RAG_document: spotify app reviews (2000 rows only to speed up mockup)

## Problem:
1. Limited computational resource “no gpu”
2. low embedding relevance.
3. Rerank LLM based is expensive and slow.

## Solution:

* Gemma:2b using ollama.
* We use “LLM keywords extraction-RAG” to retrieve relevant information for the question, then throw it to second LLM layer to formulate final answer.


### Tutorial to start the chat bot streamlit

install ollama first.

at terminal run:
> curl https://ollama.ai/install.sh | sh

> ollama run gemma:2b

> streamlit run robo.py