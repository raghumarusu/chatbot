from fastapi import FastAPI, Request
from model import generate_response
from qdrant import search_vectors
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "GPT-2 Chatbot with Qdrant"}

@app.post("/chat/")
async def chat(request: Request):
    data = await request.json()
    query = data['query']
    
    # Step 1: Search for relevant chunks from Qdrant
    # (In real usage, this would involve vectorizing the query and searching in Qdrant)
    response_chunks = search_vectors(query)
    
    # Step 2: Use GPT-2 to generate a response (append ranked results to the prompt)
    context = ' '.join(response_chunks) if response_chunks else ""
    final_query = query + context
    response = generate_response(final_query)

    return {"response": response}
