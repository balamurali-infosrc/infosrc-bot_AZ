# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import traceback
from datetime import datetime
from http import HTTPStatus

import os
import uuid
from azure.cosmos import CosmosClient


from aiohttp import web
from aiohttp.web import Request, Response

from botbuilder.core import TurnContext
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication
from botbuilder.schema import Activity, ActivityTypes

from config import DefaultConfig
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# RAG IMPORTS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# Load environment variables
load_dotenv()

CONFIG = DefaultConfig()

# Create adapter
ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(CONFIG))

# Global error handler (unchanged)
async def on_error(context: TurnContext, error: Exception):
    print(f"\n[on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()
    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity("To continue to run this bot, please fix the bot source code.")
    if context.activity.channel_id == "emulator":
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.utcnow(),
            type=ActivityTypes.trace,
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error",
        )
        await context.send_activity(trace_activity)

ADAPTER.on_turn_error = on_error

# RAG SETUP
print("Loading Rules PDFs and building knowledge base...")

pdf_folder = "Rules"
all_documents = []

# Load all PDFs
for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file_name)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)

# Splits the text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(all_documents)

# Use all docs as texts
texts = docs

# Embeddings + Chroma
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# db = Chroma.from_documents(
#     texts, embeddings,
#     collection_name="rules_collection",
#     persist_directory="./chroma_db"
# )
# db.persist()

# retriever = db.as_retriever(search_kwargs={"k": 3})
COSMOS_URI = os.getenv("COSMOS_URI")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DB_NAME = os.getenv("COSMOS_DB_NAME")
CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME")

client = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)
database = client.get_database_client(DB_NAME)
container = database.get_container_client(CONTAINER_NAME)

def store_to_cosmos(texts, embeddings):
    for text, emb in zip(texts, embeddings):
        item = {
            "id": str(uuid.uuid4()),
            "content": text,
            "embedding": emb   # Stored as vector array
        }
        container.create_item(item)

store_to_cosmos(texts, embeddings)

class CosmosRetriever:
    def __init__(self, container, k=3):
        self.container = container
        self.k = k

    def search(self, query_embedding):
        results = self.container.query_items(
            """
            SELECT TOP @k c.content, VectorDistance(c.embedding, @vec) AS score
            FROM c
            ORDER BY score
            """,
            parameters=[
                {"name": "@k", "value": self.k},
                {"name": "@vec", "value": query_embedding}
            ],
            enable_cross_partition_query=True,
        )
        return list(results)
    
query_emb = embeddings.embed_query("rules for tobacco")

retriever = CosmosRetriever(container, k=3)
docs = retriever.search(query_emb)

for doc in docs:
    print(doc["content"], " | score:", doc["score"])

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Your RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

print("RAG system ready! Bot can now answer from PDFs.")

# LLM BOT
class MyLLMBot:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_history = []

    async def on_turn(self, turn_context: TurnContext):
        if turn_context.activity.type == ActivityTypes.message:
            await self.on_message_activity(turn_context)

    async def on_message_activity(self, turn_context: TurnContext):
        user_message = turn_context.activity.text.strip()
        self.conversation_history.append({"role": "user", "content": user_message})

        # USE RAG FOR QUESTIONS ABOUT RULES
        try:
            # Run RAG
            result = qa_chain.invoke({"query": user_message})
            reply_text = result["result"]
            sources = result["source_documents"]

            # Add sources to reply
            source_text = ""
            seen = set()
            for doc in sources:
                src = doc.metadata['source']
                page = doc.metadata['page'] + 1
                filename = os.path.basename(src)
                key = (filename, page)
                if key not in seen:
                    seen.add(key)
                    source_text += f"\n• {filename} (Page {page})"

            full_reply = f"{reply_text}\n\n**Sources:**{source_text}" if source_text else reply_text

            self.conversation_history.append({"role": "assistant", "content": full_reply})
            await turn_context.send_activity(full_reply)

        except Exception as e:
            print(f"RAG Error: {e}")
            # Fallback to normal GPT if RAG fails
            try:
                completion = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        *self.conversation_history
                    ],
                    max_tokens=400,
                    temperature=0.7,
                )
                reply_text = completion.choices[0].message.content.strip()
                self.conversation_history.append({"role": "assistant", "content": reply_text})
                await turn_context.send_activity(reply_text)
            except Exception as e2:
                await turn_context.send_activity("Sorry, I couldn't respond right now.")

# Create the bot instance
BOT = MyLLMBot()


# Listen for incoming requests
async def messages(req: Request) -> Response:
    return await ADAPTER.process(req, BOT)

APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    try:
        web.run_app(APP, host="localhost", port=CONFIG.PORT)
    except Exception as error:
        raise error