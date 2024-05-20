from openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
#from langchain.vectorstores import FAISS, Pinecone as PineconeLangchang  # or another vector store implementation
from langchain.docstore.document import Document
from langchain_community.embeddings.openai import OpenAIEmbeddings

from pydantic import BaseModel
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os
import pickle

import pinecone
from langchain_pinecone import PineconeVectorStore

import logging

from LocalMetadata import LocalMetadata

DATA_DIR = "../input/"

class MyEmbeddings(OpenAIEmbeddings):
    #embedding_list: list = []

    def __init__(self):
        # BaseModel.__init__(self)  # Initialize BaseModelMixin
        OpenAIEmbeddings.__init__(self, openai_api_key="nah...")  # Initialize OpenAIEmbeddings
        #self.embedding_list = list()

    def get_embedding(self, text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        text = text.replace("\n", " ")

        emb = client.embeddings.create(input=[text], model=model).data[0].embedding
        #self.embedding_list.append(emb)
        return emb

    def embed_documents(self, text):
        return [self.get_embedding(t) for t in text]















def loadContextFromFiles(dir: str, metadata:LocalMetadata) -> list:
    previously_uploaded_files = metadata.get_previously_uploaded()

    all_files = os.listdir(dir)

    # different extensions
    loaded_documents = dict()

    filetypes = {".txt": TextLoader,
                 ".pdf": PDFPlumberLoader}

    for ext in filetypes.keys():
        files = [f for f in all_files if f.lower().endswith(ext)]


        for file in files:
            if not file in loaded_documents.keys() and not file in previously_uploaded_files:
                logging.log(logging.INFO, f"processing {file}...")
                # Create the full path to the PDF file
                full_path = os.path.join(dir, file)
                loader = filetypes[ext](full_path)
                document = None
                if ext == '.pdf': document = loader.load()
                if ext == '.txt': document = loader.load()   # quickfix - different outputs for loaders :((
                # Load the documents from the PDF and append to the list

                loaded_documents[file] = document
                metadata.set_previously_uploaded(file)




    return loaded_documents



def caluclateEmbeddingsOfContext( loaded_documents):
    """Calculates embeddings from loaded documents and stores in a vector store"""


    PINECONE_API_KEY = "2bf10b7e-86c5-4477-89bf-02628836beda"
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    pine = pinecone.Pinecone(api_key="b3fdd559-915c-46a2-833e-bfddd537fdb4")
    index = pine.Index("kafka-rag-index")

    #  Generate  embeddings for the text data


    embeddings = MyEmbeddings()


    vectors_to_insert = list()
    for i, doc in enumerate(loaded_documents):
        embedding = embeddings.get_embedding(doc.page_content)
        vectors_to_insert.append({
            "id": f"{str.rsplit(doc.metadata["source"],"/",1)[1]}_{i}",
            "values": embedding,
            "metadata": {
                "text":  doc.page_content,
                "title": doc.metadata["source"]
            }
        })
    # Upsert vectors into the Pinecone index, should skip repeats so as long as i include the name of the source in the id - it should prevent double entries
    index.upsert(vectors=vectors_to_insert)


    #need to return langchain type so we can chain it
    vectorstore = PineconeVectorStore(index_name="kafka-rag-index", embedding=embeddings, namespace="kafka-rag-index",
                                   pinecone_api_key=PINECONE_API_KEY)

    return vectorstore


def storeEmbeddingsOfContextInPineconeUsingLangchain( loaded_documents):
    """Calculates embeddings from loaded documents and stores in a vector store,
     the docs should be filtered for duplicates by now"""

    PINECONE_API_KEY = "2bf10b7e-86c5-4477-89bf-02628836beda"
    embeddings = MyEmbeddings()

    if loaded_documents!=None and len(loaded_documents) > 0:
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        #texts = [doc.page_content for doc in loaded_documents]
        #documents = [Document(page_content=text) for text in texts]


        pinecone = PineconeVectorStore.from_documents(
                loaded_documents, embeddings, index_name="kafka-rag-index", namespace="kafka-rag-index")



    else:
    #  Generate  embeddings for the text data

        pinecone = PineconeVectorStore(index_name="kafka-rag-index", embedding=embeddings, namespace="kafka-rag-index",
                                       pinecone_api_key=PINECONE_API_KEY)

    return pinecone




def splitContext(chunk_size, chunk_overlap, context):
    result = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    a = list(context.values())
    # debugging
    #ka0 = list(context.keys())[0]
    #ka1 = list(context.keys())[1]
    #x = context[ka0]
    #y = context[ka1]


    b =   [item for sublist in a for item in sublist]

    return text_splitter.split_documents(b)



def getPrompt(template=None):
    if template==None:
        template = """
            Answer the question based on the context below. If you can't 
            answer the question, reply "I don't know".

            Context: {context}

            Question: {question}
            """
    prompt = ChatPromptTemplate.from_template(template)

    return prompt

def getModel():
    model = ChatOpenAI(temperature=0.0, base_url="http://localhost:1234/v1", api_key="not-needed")
    return model


def getParser():

    return StrOutputParser()

def getChain(vector_store,prompt,model,parser):

    chain = (
        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    return chain
