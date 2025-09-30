import os
import re
from langchain_community.chat_models import ChatPerplexity
from retriever import build_retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

title2id = dict()

def format_docs(docs):
    global title2id
    title2id = dict()
    for doc in docs:
        movie_id = doc.metadata['movie_id']
        title = doc.metadata['title']
        title2id[title] = movie_id
    return '\n\n'.join(doc.page_content for doc in docs)

def create_session(temperature=0):
    prompt = ChatPromptTemplate.from_template("""
    You are a movie recommendation assistant. Recommend three movies from below list, choosing 3 movies that best match user's request.
    For each chosen movie, give:
    1) title (do not involve release date)
    2) a 2-3 sentence why it matches

    Provide a clear and concise answer. Do not provide footnote with "[]". Only recommend movies from the below context: {context}
    
    Question: {question}
    Answer:
    """)
    load_dotenv()
    model = ChatPerplexity(model="sonar", pplx_api_key=os.environ.get("PPLX_API_KEY"), temperature=temperature)
    retriever = build_retriever()
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain

def invoke_query(rag_chain, query):
    response = rag_chain.invoke(query)
    titles = re.findall(r"\*\*(.*?)\*\*", response)
    ids = [title2id.get(title) for title in titles]
    ids = [id for id in ids if id is not None]
    ids = list(set(ids))
    return response, ids

if __name__ == "__main__":
    rag_chain = create_session()
    response, ids = invoke_query(rag_chain, "recommend me some movies about heros. I especially like actions.")
    print("Answer:\n", response)
    print()
    print()
    print(ids)
    print("=" * 50)

    response, ids = invoke_query(rag_chain, "recommend me some horror movies where monster comes out")
    print("Answer:\n", response)
    print()
    print()
    print(ids)

    response, ids = invoke_query(rag_chain, "recommend me some funny movie where old man is main character")
    print("Answer:\n", response)
    print()
    print()
    print(ids)