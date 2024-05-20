from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

# backend.py
from fastapi import FastAPI, Form, HTTPException, Request, Depends, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import llm
import os
from functools import lru_cache
import asyncio
import uuid

from LocalMetadata import LocalMetadata

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific origins here
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Simple message store to keep a list of messages
chat_history = []
async_results = {}


# Define the Pydantic model
class Query(BaseModel):
    query: str  # The expected field in the JSON body


@app.get("/get_result/{task_id}", response_class=HTMLResponse)
async def get_result(task_id: str):
    result = async_results.get(task_id, "Task is still in progress...")
    return f"<p>{result}</p>"


async def getResponseFromLLM(query):
    global vector_store
    prompt = llm.getPrompt()
    model = llm.getModel()
    parser = llm.getParser()
    #vector_store = Depends(prepareContext)
    chain = llm.getChain(vector_store,prompt,model,parser)
    response = chain.invoke(query)


    return response


@app.get("/output/")
async def get_output():
    #prompt = llm.getPrompt()

    #vector_store = Depends(prepareContext)

    #response = chain.invoke(QUERY)
    messages = "some messages"
    return JSONResponse(content={"messages": "<br>".join(messages)})


@app.post("/input/", response_class=HTMLResponse)
async def post_message(query: str = Form(...)):
    # Extract the message and store it
    # Extract the "message" field

    query_id = 0
    response_id = 1

    chat_history.append((query, query_id))

    task_id = str(uuid.uuid4())
    response = await getResponseFromLLM(query)

    #test_results = vector_store.similarity_search_with_score(query=query, k=3) # test


    response_html = f"""
         <p class="chat-style">  {response} </p>
    """
    return response_html


@lru_cache()  # Ensures single instance (cached)
def prepareContext(data_dir):
    #metadata = LocalMetadata()
    with LocalMetadata() as metadata:
        loaded_context = llm.loadContextFromFiles(data_dir, metadata)
        docs = llm.splitContext(chunk_size=1000, chunk_overlap=20, context=loaded_context)
        vector_store = llm.storeEmbeddingsOfContextInPineconeUsingLangchain(docs)
        metadata.save_local_metadata()

    return vector_store

#PINECONE_API_KEY = "PUT YOUR KEY"
#os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


global vector_store
vector_store = prepareContext(llm.DATA_DIR)

if __name__ == "__main__":
    #vector_store = prepareContext(DATA_DIR)

    uvicorn.run(app, host="0.0.0.0", port=8000)
