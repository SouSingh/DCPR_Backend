from fastapi import FastAPI, File, UploadFile, HTTPException
import openai
from dotenv import load_dotenv
import os
import json
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import weaviate
import uvicorn

load_dotenv()

app = FastAPI()

api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)

client = weaviate.Client(
  url=WEAVIATE_URL,
  auth_client_secret=auth_config
)


def search_and_query():
    blogs = SimpleDirectoryReader("./Data").load_data()
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="DCPR")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(blogs, storage_context=storage_context)
    return "Done Embeddings"

def Quert(ask):
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="DCPR")
    loaded_index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query(ask)
    return response


def contract_analysis_w_fact_checking(text):

    if not text:
        raise HTTPException(
            status_code=400, detail="Text field is required in the input data.")
    print("done 1")
    # Perform contract analysis using Quert (assuming Quert is a class or function)
    quert_instance = Quert(text)
    
    # Extract relevant information from the Quert response
    if quert_instance.response:
        contract_results = [{
            "LLM Response": quert_instance.response,
            "Source_node": [{
                "Page_number": key_point.node.metadata.get('page_label', ''),
                "File_Name": key_point.node.metadata.get('file_name', ''),
                "Text": key_point.node.text,
                "Start_Char": key_point.node.start_char_idx,
                "End_Char": key_point.node.end_char_idx,
                "Score_Matching": key_point.score
            } for key_point in quert_instance.source_nodes]
        }]
    else:
        contract_results = []
    
    # Return a standardized response
    return {"status": "success", "message": "Contract analysis successful", "model_response": contract_results}


@app.post("/embedd")
async def predict():
    try: 
        dor = search_and_query()
        return {"user_content": dor}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(data: dict):
    try:
        messages = data.get("messages", [])
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
        out = contract_analysis_w_fact_checking(user_message)
        if user_message:
            return {"user_content": out}
        else:
            raise HTTPException(
                status_code=400, detail="User message not found in input.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/")
def read_root():
    return {"Hello": "World"}
