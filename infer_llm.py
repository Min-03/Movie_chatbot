import os
import re
from retriever import build_retriever 
from langchain_community.chat_models import ChatPerplexity
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Annotated

class MovieRec(BaseModel):
    movie_id: str = Field(description="Id of the movie, which is included in metadata")
    title: str = Field(description="Title of the movie")
    reason: str = Field(description="Reasons for recommeding the movie, should be 2~3 sentences")

class MovieResponse(BaseModel):
    results: Annotated[List[MovieRec], Field(min_items=3, max_items=3)]

def format_docs(docs):
    formatted = []
    for doc in docs:
        formatted.append(f"Movie ID:{doc.metadata['movie_id']}\nContent: {doc.page_content}")
    return "\n\n".join(formatted)

def create_session(temperature=0):
    prompt = ChatPromptTemplate.from_template("""
    You are a movie recommendation assistant. Recommend exactly three different movies based on the given context.

    Only recommend movies explicitly included in the context.

    Return results following this JSON schema:
    {format}

    Context:
    {context}

    Question: {question}
    """)
    load_dotenv()
    model = ChatPerplexity(model="sonar-pro", pplx_api_key=os.environ.get("PPLX_API_KEY"), temperature=temperature)
    retriever = build_retriever(search_num=20)
    reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=reranker, top_n=5)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    parser = PydanticOutputParser(pydantic_object=MovieResponse)
    prompt = prompt.partial(format=parser.get_format_instructions())
    
    rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    return rag_chain

def invoke_query(rag_chain, query):
    response_all = rag_chain.invoke(query)
    response, ids, titles = [], [], []
    for rec in response_all.results:
        response.append(rec.reason)
        ids.append(rec.movie_id)
        titles.append(rec.title)
    return response, ids, titles

if __name__ == "__main__":
    rag_chain = create_session()
    response, ids, titles = invoke_query(rag_chain, "recommend me some movies about heros. I especially like actions.")
    print("Answer:\n", response)
    print()
    print()
    print(ids)
    print()
    print()
    print(titles)
    print("=" * 50)

    response, ids, titles = invoke_query(rag_chain, "recommend me some horror movies where monster comes out")
    print("Answer:\n", response)
    print()
    print()
    print(ids)
    print()
    print()
    print(titles)
    print("=" * 50)


    response, ids, titles = invoke_query(rag_chain, "recommend me some funny movie where old man is main character")
    print("Answer:\n", response)
    print()
    print()
    print(ids)
    print()
    print()
    print(titles)