import os
import uuid
import fitz  # PyMuPDF
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="InsightPDF: Semantic Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Qdrant Client (In-Memory)
qdrant_client = QdrantClient(":memory:")
COLLECTION_NAME = "pdf_documents"

# Initialize Embedding Model
# This downloads a ~80MB model on the first run
model = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_SIZE = 384 

# Create collection (Using modern non-deprecated check)
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

class SearchResult(BaseModel):
    text: str
    pdf_name: str
    page: int
    score: float

@app.post("/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    total_chunks = 0
    try:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue
                
            content = await file.read()
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            
            points = []
            for page_num, page in enumerate(pdf_doc):
                text = page.get_text("text").strip()
                if len(text) < 15: # Skip very short/empty pages
                    continue
                
                # Generate embedding
                embedding = model.encode(text).tolist()
                
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": text[:2000],
                        "pdf_name": file.filename,
                        "page": page_num + 1
                    }
                ))
            
            if points:
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                total_chunks += len(points)
        
        return {"message": f"Successfully ingested {len(files)} files ({total_chunks} pages)."}
    except Exception as e:
        logger.error(f"Ingestion Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=List[SearchResult])
async def search(query: str):
    try:
        if not query:
            return []

        # Convert query to embedding
        query_vector = model.encode(query).tolist()
        
        # We try the two most common search methods in Qdrant-Client to ensure compatibility
        try:
            # Method 1: Modern query_points (v1.10+)
            response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=5
            )
            search_results = response.points
        except (AttributeError, Exception):
            # Method 2: Legacy search fallback
            logger.info("query_points not found, falling back to legacy search")
            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=5
            )
        
        results = [
            SearchResult(
                text=hit.payload["text"],
                pdf_name=hit.payload["pdf_name"],
                page=hit.payload["page"],
                score=float(hit.score)
            )
            for hit in search_results
        ]
        
        return results
    except Exception as e:
        logger.error(f"Search Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    base_path = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_path, "index.html")
    with open(index_path, "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)