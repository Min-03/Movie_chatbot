from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_docs(csv_dir, save_dir="/data/chroma", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    docs = []
    df = pd.read_csv(csv_dir)

    for _, row in df.iterrows():
        movie_id = row.get("rotten_tomatoes_link")
        title = row.get('movie_title')
        info = row.get("movie_info")
        genre = row.get("genres")
        runtime = row.get("runtime")
        # director = row.get("directors")
        # actor = row.get("actors")
        # date = row.get("original_release_date")

        contents = [f"Title: {title}"]
        contents.append(f"Genre: {genre}")
        contents.append(f"Runtime(minutes) : {runtime}m")
        contents.append(f"Description of the movie: {info}")

        contents = "\n".join(contents)
        metadata = {"movie_id": str(movie_id), "title": title}
        docs.append(Document(page_content=contents, metadata=metadata))

    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embedding_func = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding_func, persist_directory=save_dir)
    vectordb.persist()

def build_retriever(load_dir="/data/chroma", embedding_model="sentence-transformers/all-MiniLM-L6-v2", search_num=5):
    embedding_func = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma(persist_directory=load_dir, embedding_function=embedding_func)
    retriever = vectordb.as_retriever(seach_kwargs={"k": search_num})

    return retriever
    
def test_retriever(query):
    retriever = build_retriever()
    docs = retriever.invoke(query)
    for doc in docs:
        print(doc)
        print("=" * 50)

if __name__ == "__main__":
    # build_docs(csv_dir="data/movies.csv")
    test_retriever("recommend me some movies about heros. I especially like actions.")