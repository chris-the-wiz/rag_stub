# Rag Stub

Everybody and their mother do LLM + RAG personalized agents these days. So do I.

Langchain + pinecone embeddings + fastAPI backend + htmx frontend


Working and ready to be containerized. 

##USAGE:

1. Run LM Studio and load  your fav LLM model and embbedings model.
2. Set your environmental variables
3. Run  backend.py
4. Open browser on 127.0.0.1:8000
5. ???
6. Profit

At the first run it needs to generate the embeddings. This takes time. 


## ENV VARIABLES:

 - POSTGRES_CONNECTION_STRING=postgresql://<name>:<pass>@<ip>:<port>/<db>
 - PINECONE_API_KEY=get-it-from-pinecone-dashboard
 - DATA_DIR="/where/do/you/keep/your/input/files/to/generate/embeddings"
 - STORAGE=postgres_langchain | pinecone  |  postgres

## TODO:
 - dockerfile
 - cleanup
 - oolama interface
 
 
