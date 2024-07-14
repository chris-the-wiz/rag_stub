from pydantic import BaseModel
# from typing import Dict, Any
import uvicorn

# backend.py
from fastapi import FastAPI, Form, # HTTPException, Request, Depends, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import llm
import os
from functools import lru_cache
import asyncio
# import uuid
import asyncpg
from pgvector.asyncpg import register_vector

from LocalMetadata import LocalMetadata

import math









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



# connection string for postgres
connection_string = os.environ['POSTGRES_CONNECTION_STRING']


PINECONE_API_KEY = os.environ['PINECONE_API_KEY']

# Define the Pydantic model
class Query(BaseModel):
    query: str  # The expected field in the JSON body



# Route to serve index.html
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("frontend/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

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
    response = str()
    query_id = 0
    response_id = 1

    chat_history.append((query, query_id))

    # Get the response

    if os.environ["STORAGE"] == "pinecone":

        response = await getResponseFromLLM(query)
        #test_results = vector_store.similarity_search_with_score(query=query, k=3) # test

    elif os.environ["STORAGE"] == "postgres_langchain":
        response = await getResponseFromLLM(query)


    #dirty experimental stuff below. i mean... even more experimental than the rest of this code :>
    elif os.environ["STORAGE"] == "postgres":

        # Step 1: Get documents related to the user input from database
        related_docs = await llm.get_top3_similar_docs(llm.get_embeddings(query), conn)

        # Step 2: Get completion from OpenAI API
        # Set system message to help set appropriate tone and context for model

        system_message = f"""
                Answer the question based on the context below.
                If you can't answer the question, reply "I don't know".

                Context: {related_docs[0], related_docs[1], related_docs[2]}

                """


        # Prepare messages to pass to model
        # We use a delimiter to help the model understand the where the user_input starts and ends
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{query}"},

        ]



        response = llm.get_completion_from_messages(messages, temperature=0.0, max_tokens=500)


    response_html = f"""
        <p class="chat-style">  {response} </p>
    """


    return response_html


@lru_cache()  # Ensures single instance (cached)
async def prepare_context(data_dir, storage):

    if storage == "pinecone":
        with LocalMetadata() as metadata:
            loaded_context = llm.loadContextFromFiles(data_dir, metadata)
            docs = llm.splitContext(chunk_size=1000, chunk_overlap=20, context=loaded_context)
            vector_store = llm.storeEmbeddingsOfContextInPineconeUsingLangchain(docs)
            metadata.save_local_metadata()

    elif storage == "postgres_langchain":
        with LocalMetadata() as metadata:
            loaded_context = llm.loadContextFromFiles(data_dir, metadata)
            docs = llm.splitContext(chunk_size=1000, chunk_overlap=20, context=loaded_context)
            vector_store = llm.storeEmbeddingsOfContextInPostgresUsingLangchain(docs)
            metadata.save_local_metadata()





    elif storage == "postgres":
        global conn
        conn = await asyncpg.connect(connection_string)
        await register_vector(conn)

        await conn.execute('CREATE TABLE IF NOT EXISTS embeddings ('
                           'id bigserial primary key,'
                           'title text,'
                           'content text,'
                           'tokens integer,'
                           'embedding vector(768));'
        )




        with LocalMetadata() as metadata:
            loaded_context = llm.loadContextFromFiles(data_dir, metadata)
            docs = llm.splitContext(chunk_size=1000, chunk_overlap=20, context=loaded_context)
            vector_store = llm.caluclateEmbeddingsOfContext(docs)
            await conn.executemany(
                'INSERT INTO embeddings (title, content, tokens, embedding) VALUES ($1, $2, $3, $4);',
                [(doc["id"], doc["metadata"]["text"], doc["metadata"]["tokens"], doc["values"]) for doc in vector_store]
            )

            # Create an index on the data for faster retrieval

            # calculate the index parameters according to best practices
            num_records = len(vector_store)
            num_lists = num_records / 1000
            if num_lists < 10:
                num_lists = 10
            if num_records > 1000000:
                num_lists = math.sqrt(num_records)

            await conn.execute(
                'CREATE INDEX IF NOT EXISTS embeddings_idx '
                'ON embeddings USING ivfflat(embedding vector_cosine_ops) '
                f'WITH (lists = {num_lists});')



            metadata.save_local_metadata()



    return vector_store




#global vector_store
# vector_store = asyncio.run(prepare_context(llm.DATA_DIR))

if __name__ == "__main__":

    vector_store = asyncio.run(prepare_context(llm.DATA_DIR, storage=os.environ["STORAGE"]))
    uvicorn.run(app, host="0.0.0.0", port=8000)
