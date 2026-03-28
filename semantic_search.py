import chromadb
from sentence_transformers import SentenceTransformer

# 1. Setup (Same 'Brain' and 'Storage' as before)
print("Loading Semantic Engine...")
model = SentenceTransformer('clip-ViT-B-32', device='cpu')
client = chromadb.PersistentClient(path="./data/vector_db")
collection = client.get_collection(name="multimodal_video_index")

def semantic_search(query, n_results=5):
    # 2. Vectorize the User Query
    # This turns "cheese" into a 512-dimension coordinate
    query_vector = model.encode(query).tolist()

    # 3. Perform the Vector Search
    # Chroma finds the closest 'neighbors' in the latent space
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )

    print(f"\nTop {n_results} Semantic Matches for: '{query}'")
    print("-" * 60)

    # 4. Parse and Display
    for i in range(len(results['ids'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        # Distance is '1 - Cosine Similarity'. Lower is better/closer.
        score = 1 - results['distances'][0][i] 
        
        print(f"Score: {score:.2f} | {meta['video_id']} @ {meta['timestamp']:.2f}s")
        print(f"Context: {doc[:100]}...") # Print snippet of the fused data
        print("-" * 30)
def semantic_search_with_boost(query, n_results=5):
    query_vector = model.encode(query).tolist()
    
    # 1. Expand search to Top 20 (Recall Phase)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=20,
        include=['documents', 'metadatas', 'distances']
    )

    scored_results = []
    for i in range(len(results['ids'][0])):
        doc = results['documents'][0][i].lower()
        meta = results['metadatas'][0][i]
        
        # Base Semantic Score
        base_score = 1 - results['distances'][0][i]
        
        # 2. THE BOOST LOGIC (Expert Layer)
        # If the actual word exists in the OCR or Visuals, we apply a multiplier.
        boost = 1.0
        if query.lower() in doc:
            # Significant boost for exact matches in the text
            boost = 1.5 
        
        # Extra boost if it's explicitly in the "Text:" (OCR) section
        if f"text: {query.lower()}" in doc:
            boost = 1.8

        final_score = base_score * boost
        
        scored_results.append({
            "id": results['ids'][0][i],
            "score": final_score,
            "meta": meta,
            "doc": doc
        })

    # 3. Re-sort based on boosted scores
    scored_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)

    # Display top N
    print(f"\nRE-RANKED Matches for: '{query}'")
    for r in scored_results[:n_results]:
        print(f"Boosted Score: {r['score']:.2f} | {r['meta']['video_id']} @ {r['meta']['timestamp']:.2f}s")
        print(f"Context: {r['doc'][:100]}...")
        print("-" * 30)

def search(query, n_results=5):
    query_norm = query.lower().strip()
    query_vec = model.encode(query_norm).tolist()
    
    # 1. Pull more candidates than we need (Recall)
    raw = collection.query(query_embeddings=[query_vec], n_results=20)

    scored = []
    for i in range(len(raw['ids'][0])):
        doc = raw['documents'][0][i].lower()
        meta = raw['metadatas'][0][i]
        
        # Base Similarity (0.0 to 1.0)
        score = 1 - raw['distances'][0][i]
        
        # 2. Additive Boost (Logic Over Magic)
        # If the word exists literally, we add a flat constant.
        if query_norm in doc:
            score += 0.5 
            
        # Specific field boosting
        if f"audio: " in doc and query_norm in doc.split("audio: ")[1]:
            score += 0.2
        if f"text: " in doc and query_norm in doc.split("text: ")[1]:
            score += 0.4

        scored.append({"score": score, "meta": meta, "doc": doc})

    # 3. Sort by the new score
    scored = sorted(scored, key=lambda x: x['score'], reverse=True)

    for r in scored[:n_results]:
        print(f"[{r['score']:.2f}] {r['meta']['video_id']} @ {r['meta']['timestamp']}s")
def hybrid_search(query, n_results=5):
    query_norm = query.lower().strip()
    
    # 1. GET ALL DATA (Broad Recall)
    # We pull a larger set to re-rank
    all_docs = collection.get(include=['documents', 'metadatas', 'embeddings'])
    
    scored_results = []
    
    # 2. EVALUATE EVERY DOC
    for i in range(len(all_docs['ids'])):
        doc_text = all_docs['documents'][i].lower()
        meta = all_docs['metadatas'][i]
        
        # LITERAL SCORE (Fuzzy/Exact)
        # We give this a massive weight. If the word is there, it's a 1.0.
        literal_score = 1.0 if query_norm in doc_text else 0.0
        
        # SEMANTIC SCORE (The "Helper")
        # Use the model to see how 'close' the meaning is (0.0 to 1.0)
        query_vec = model.encode(query_norm)
        doc_vec = all_docs['embeddings'][i]
        
        # Cosine Similarity math
        import numpy as np
        semantic_score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        
        # FINAL RANKING: Literal is the anchor, Semantic is the tie-breaker
        # A literal match (1.0) will always beat a 0.8 semantic match.
        final_score = (literal_score * 2.0) + semantic_score
        
        if final_score > 0.3: # Filter out garbage
            scored_results.append({
                "score": final_score,
                "video": meta['video_id'],
                "time": meta['timestamp'],
                "text": doc_text
            })

    # Sort and return
    scored_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)
    for r in scored_results[:n_results]:
        print(f"[{r['score']:.2f}] {r['video']} @ {r['time']}s")
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter search term (or 'exit'): ")
        if user_query.lower() == 'exit': break
        hybrid_search(user_query)