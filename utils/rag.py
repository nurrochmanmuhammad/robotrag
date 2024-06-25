from pathlib import Path
import csv

from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
    SimpleDirectoryReader
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.readers.file import CSVReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from chromadb import EmbeddingFunction, Documents, Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

class LlamaIndexEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, ef: BaseEmbedding):
        self.ef = ef

    def __call__(self, input: Documents) -> Embeddings:
        return [node.embedding for node in self.ef([TextNode(text=doc) for doc in input])]


class Ingestion:
    def __init__(self, path='data/SPOTIFY_REVIEWS.csv', embedding_model="BAAI/bge-small-en-v1.5", rowlimit=2000):
        self.path = path
        self.embedding_model = embedding_model
        self.rowlimit = rowlimit

    def run(self):
        print('Data ingestion')
        page_content = []

        try:
            with open(self.path, newline='', encoding="utf8") as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                i = 0
                for row in csvreader:
                    if i > 0:
                        content = f'\n{row[4]}\n'
                        meta = {"review_id": row[1], 'author_name': row[3]}
                        page_content.append(content)
                    i += 1

                    if self.rowlimit and i >= self.rowlimit:
                        break

            self.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
            client = chromadb.Client()
            self.collection = client.get_or_create_collection("test_collection", embedding_function=LlamaIndexEmbeddingAdapter(self.embed_model))
            page_content = list(set(page_content))
            ids = [str(i) for i in range(len(page_content))]
            self.collection.add(ids=ids, documents=page_content)

        except Exception as e:
            print(f"Error during ingestion: {e}")
            # Handle the exception as appropriate for your application


class RAGBot:
    def __init__(self, collection, MODEL_NAME="gemma:2b", REQUEST_TIMEOUT=60):
        self.llm = Ollama(model=MODEL_NAME, request_timeout=REQUEST_TIMEOUT)
        self.collection = collection

    def qna(self, question, RAG_KEYWORD=30,get_query=False):
        '''
        Performs Q&A using RAG method.

        Args:
            question (str): The question to answer.
            RAG_KEYWORD (int): Number of relevant keywords to fetch.

        Returns:
            str: The response to the question.
        '''

        try:
            print('inference process')
            response1 = self.llm.complete(f'''
            give me {RAG_KEYWORD} list relevant keywords from this question:

            {question}
            ''')

            query = self.collection.embed_model.get_text_embedding(f'question: {question} \n' + response1.text)
            results = self.collection.collection.query(query, n_results=20)['documents'][0]
            info_rag = '\n'.join([c for c in results])

            st = f'''
            I am a Spotify developer.
            You are helping me to answer questions based on Spotify app reviews.
            Answer this question: {question}

            App reviews:
            {info_rag}
            '''

            response2 = self.llm.complete(st)
            if get_query:
                return st,response2.text
            else:
                return response2.text

        except Exception as e:
            print(f"Error during Q&A: {e}")
            return None  # Or handle the error in an appropriate way