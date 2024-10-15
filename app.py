from wordllama import WordLlama
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#Add CORS middleware 
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Load the default WordLlama model
wl = WordLlama.load()


@app.get("/cluster")
def cluster_docs(doc_string : str, clusters : int):
    docs_list = doc_string.split("/")
    labels, inertia = wl.cluster(docs_list, k=clusters, max_iterations=100, tolerance=1e-4, n_init=clusters)

    clustered_docs = {}
    for i, text in zip(labels, docs_list):
        if i not in clustered_docs:
            clustered_docs[i] = []
        clustered_docs[i].append(text)
    
    return clustered_docs