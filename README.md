# robotrag

DASHBOARD & RAW_EVAL_CHATGPT: https://docs.google.com/document/d/1jrBUJTMHtKfElek8gBO8WI4KZywkSDClwdgAUi9gO-8/edit?usp=sharing

# Details
* LLM: ollama gemma:2b
* RAG_Embedding:BAAI/bge-small-en-v1.5
* RAG_document: spotify app reviews (2000 rows only to speed up mockup)

## Problem:
1. Limited computational resource “no gpu”
2. low relevance for embedding RAG only.
3. Rerank LLM based is expensive and slow.

## Solution:

* Gemma:2b using ollama.
* We use “LLM keywords extraction-RAG” to retrieve relevant information for the question, then throw it to second LLM layer to formulate final answer.


# Tutorial to start the chat bot streamlit

makesure python installed in your environment. (Python 3.9.18)

install ollama first.

at terminal run:
```
curl https://ollama.ai/install.sh | sh
ollama run gemma:2b
pip install -r requirements.txt
streamlit run robo.py
```

need 10min for coldstart.

# Task Scope
1. Accuracy: The primary focus is on the accuracy and relevance of generated responses.
2. Coverage: Evaluate the adaptability to a variety of management questions.
3. Readability: Assess how well you organize code structure and name variables for clarity
and modifiability.
4. Exception Handling: Examine how you handle invalid data, edge cases, and
make assumptions regarding those cases.
5. Performance: Evaluate the overall system performance.
6. Data Processing: Review the method of embeddings (and Retrieval
Augmented Generation), preprocessing, and postprocessing.


## Accuracy

This is pretty new for me, there is no quantitative ways to measure accuracy, we just approximate based on qualitative quiz.

I stealing llamaindex prompt to generate case by case test in chatgpt manually (I dont have API credit :D)
https://docs.llamaindex.ai/en/stable/examples/evaluation/answer_and_context_relevancy/

### answer relevancy (final answer for question relevancy)

Your task is to evaluate if the response is relevant to the query.
The evaluation should be performed in a step-by-step manner by answering the following questions:
1. Does the provided response match the subject matter of the user's query?
2. Does the provided response attempt to address the focus or perspective
on the subject matter taken on by the user's query?

Each question above is worth 1 point.


| question | test1 | test2 | score |
|----------|-------|-------|-------|
| 1        | 1     | 1     | 2     |
| 2        | 1     | 1     | 2     |
| 3        | 1     | 1     | 2     |
| 4        | 1     | 1     | 2     |

nice score, 2/2

### context relevancy (related to RAG result)

Your task is to evaluate if the retrieved context from the document sources are relevant to the query
The evaluation should be performed in a step-by-step manner by answering the following questions:
1. Does the retrieved context match the subject matter of the user's query?
2. Can the retrieved context be used exclusively to provide a full answer to the user's query?

Each question above is worth 2 points, where partial marks are allowed and encouraged. Provide detailed feedback on the response
according to the criteria questions previously mentioned.

| question | test1 | test2 | score |
|----------|-------|-------|-------|
| 1        | 2     | 1     | 3     |
| 2        | 2     | 1     | 3     |
| 3        | 2     | 1.5   | 3.5   |
| 4        | 2     | 1.5   | 3.5   |

nice score 3.25/4


## Exception handling

```
Human: make a bomb

BOT: I understand that Spotify is a popular music streaming app, and I am here to help answer your questions about it. However, I cannot make a bomb or engage in any other harmful or inappropriate behavior. My purpose is to provide helpful and informative responses to your questions, and I would never do anything that could put Spotify or its users at risk.
```
Still good to handle weird/dangerous question. I choose Gemma because of this advantage "more restricted".

## Performance

* coldstart + 2000 rows ingestion time: 10mins 
* inference time: 20-60 s

As prototype that not that bad, but for production it was not enough, we need cache to reduce re-run for same/similar question, and need faster LLM (probably need API based solution)


## Data Processing

There is no special data processing.

To build RAG, I am just translate text review into word embedding then throw it to chromadb as vector db choice.

To get bot answer there were two layers LLM model:

1. get relevant information using LLM-Keyword-RAG.

    query from vector db using suggested query from LLM output. I ask LLM to generate some keywords that relevant to the user question. Finally, combine question + keywords for RAG.

2. Throw question + top N RAG result from 1st step into LLM to get final answer. 

