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
load_dotenv()

from config import DefaultConfig
CONFIG = DefaultConfig()

# ---------------- AUTH + ADAPTER ----------------
auth = ConfigurationBotFrameworkAuthentication(CONFIG)
ADAPTER = CloudAdapter(auth)


# ---------------- ERROR HANDLER ----------------
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


# ---------------- RAG SETUP ----------------
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA

print("Building RAG DB...")

# Base directory inside container
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
PDF_PATH = os.path.join(BASE_DIR, "Rules")

# Ensure Chroma folder exists
os.makedirs(CHROMA_PATH, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="rules_collection",
    metadata={"hnsw:space": "cosine"}
)

documents = []

if os.path.exists(PDF_PATH):
    for fn in os.listdir(PDF_PATH):
        if fn.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_PATH, fn))
            documents.extend(loader.load())
else:
    print("WARNING: Rules directory not found inside container:", PDF_PATH)

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

db = Chroma(
    client=client,
    collection_name="rules_collection",
    embedding_function=embeddings,
)

if db._collection.count() == 0:
    print("Adding documents to vector database...")
    db.add_documents(docs)
else:
    print("Chroma already populated.")

retriever = db.as_retriever(search_kwargs={"k": 3})

print("RAG ready.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)


# ---------------- BOT CLASS ----------------
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

            sources = ""
            seen = set()
            for doc in rag["source_documents"]:
                filename = os.path.basename(doc.metadata["source"])
                page = doc.metadata["page"] + 1
                key = (filename, page)
                if key not in seen:
                    seen.add(key)
                    sources += f"\n• {filename} (Page {page})"

            reply = answer + ("\n\n**Sources:**" + sources if sources else "")
            await turn_context.send_activity(reply)

        except Exception as e:
            print("RAG failed:", e)
            fallback = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.history,
                temperature=0.7,
            )
            await turn_context.send_activity(fallback.choices[0].message.content)


BOT = MyLLMBot()


# ---------------- BOT ROUTE (WORKS IN AZURE) ----------------
async def messages(req: Request) -> Response:
    body = await req.json()
    auth_header = req.headers.get("Authorization", "")

    return await ADAPTER.process(
        req,
        BOT,
        body=body,
        auth_header=auth_header
    )


app = web.Application(middlewares=[aiohttp_error_middleware])
app.router.add_post("/api/messages", messages)


# ---------------- START APP ----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    web.run_app(app, host="0.0.0.0", port=port)
