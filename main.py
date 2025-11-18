# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import traceback
from datetime import datetime
from http import HTTPStatus

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

db = Chroma.from_documents(
    texts, embeddings,
    collection_name="rules_collection",
    persist_directory="./chroma_db"
)
db.persist()

retriever = db.as_retriever(search_kwargs={"k": 3})

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
async def messages(req:web.Request) -> web.Response:
    return await ADAPTER.process(req, BOT)

APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post("/api/messages", messages)
# APP.router.add_get("/", messages)


# APP.router.add_get("/", messages)
# return web.Response(text="✅ Bot App is running on Azure App Service!")
# async def handle_root(req: Request):
#     return web.Response(text="✅ Bot service is running successfully on Azure App Service!")
# APP.router.add_get("/", handle_root)

 

if __name__ == "__main__":
    try:
        web.run_app(APP, host="localhost", port=CONFIG.PORT)
        #  web.run_app(debug=True ,port=CONFIG.PORT,use_reloader=False)
        # web.run_app(APP, host="0.0.0.0", port=CONFIG.PORT, handle_signals=True)

    except Exception as error:
        raise error