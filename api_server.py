from fastapi import FastAPI, Request
from infer_llm import create_session, invoke_query

app = FastAPI()

@app.on_event('startup')
def create_chain():
    app.state.rag_chain = create_session()
    print("RAG chain is created")

@app.post("/recommend")
def recommend_movies(query: str, request : Request):
    rag_chain = request.app.state.rag_chain
    response, ids = invoke_query(rag_chain, query)
    return {"response": response, "movie_ids": ids}