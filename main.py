# Copyright (c) Microsoft.
# Licensed under MIT License.

import sys
import os
import traceback
from datetime import datetime

from aiohttp import web
from aiohttp.web import Request, Response

from botbuilder.core import TurnContext
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.integration.aiohttp import (
    CloudAdapter,
    ConfigurationBotFrameworkAuthentication,
)
from botbuilder.schema import Activity, ActivityTypes

from dotenv import load_dotenv
from config import DefaultConfig

# ---------- Load Environment ----------
load_dotenv()
CONFIG = DefaultConfig()

# ---------- Adapter & Auth ----------
auth = ConfigurationBotFrameworkAuthentication(CONFIG)
ADAPTER = CloudAdapter(auth)


# ---------- Error Handling ----------
async def on_error(context: TurnContext, error: Exception):
    print(f"\n[ERROR] {error}", file=sys.stderr)
    traceback.print_exc()

    await context.send_activity("The bot encountered an error.")
    await context.send_activity("Please fix the bot source code.")

    if context.activity.channel_id == "emulator":
        trace = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.utcnow(),
            type=ActivityTypes.trace,
            value=str(error),
            value_type="https://www.botframework.com/schemas/error",
        )
        await context.send_activity(trace)


ADAPTER.on_turn_error = on_error

# ---------- Build RAG database (DuckDB — container-safe) ----------

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
print("Building RAG DB...")

# Create modern Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Create / reuse collection (required metadata)
collection = client.get_or_create_collection(
    name="rules_collection",
    metadata={"hnsw:space": "cosine"}
)

# Load PDFs
pdf_folder = "Rules"
documents = []

for fn in os.listdir(pdf_folder):
    if fn.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, fn))
        documents.extend(loader.load())

# Split texts
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Use LangChain wrapper with the new client
db = Chroma(
    client=client,
    collection_name="rules_collection",
    embedding_function=embeddings,
)

# Add docs only once
if db._collection.count() == 0:
    print("Adding documents to vector database...")
    db.add_documents(docs)

retriever = db.as_retriever(search_kwargs={"k": 3})

print("RAG loaded successfully.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)

print("RAG system ready.")

# -------------------------------------------------------------------------
# --------------------------- BOT CLASS -----------------------------------
# -------------------------------------------------------------------------

from openai import AsyncOpenAI


class MyLLMBot:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history = []

    async def on_turn(self, turn_context: TurnContext):
        if turn_context.activity.type == ActivityTypes.message:
            await self.on_message(turn_context)

    async def on_message(self, turn_context: TurnContext):
        user_text = turn_context.activity.text.strip()
        self.history.append({"role": "user", "content": user_text})

        try:
            rag = qa_chain.invoke({"query": user_text})
            answer = rag["result"]

            # Build source list
            seen = set()
            sources = ""
            for doc in rag["source_documents"]:
                filename = os.path.basename(doc.metadata["source"])
                page = doc.metadata["page"] + 1
                key = (filename, page)

                if key not in seen:
                    seen.add(key)
                    sources += f"\n• {filename} (Page {page})"

            full_reply = answer + ("\n\n**Sources:**" + sources if sources else "")
            await turn_context.send_activity(full_reply)

        except Exception as e:
            print("RAG failure:", e)
            fallback = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.history,
                temperature=0.6,
                max_tokens=300,
            )
            reply = fallback.choices[0].message.content
            await turn_context.send_activity(reply)


BOT = MyLLMBot()

# -------------------------------------------------------------------------
# ------------------------ AIOHTTP ROUTING --------------------------------
# -------------------------------------------------------------------------


async def messages(req: Request) -> Response:
    """
    This MUST return an aiohttp Response
    or Bot Framework will throw 404 / 503.
    """
    return await ADAPTER.process(req, BOT)


app = web.Application(middlewares=[aiohttp_error_middleware])
app.router.add_post("/api/messages", messages)

# -------------------------------------------------------------------------
# -------------------------- APP START ------------------------------------
# -------------------------------------------------------------------------

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=CONFIG.PORT)
