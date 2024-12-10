from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_search import StreamEmbeddingManager, StreamRAGManager

app = FastAPI(
    title="Satori LLM"
)

class QueryRequest(BaseModel):
    user_query: str
    top_n: int = 5
    
class ExplainRequest(BaseModel):
    user_query: str
    streams: list

eb_manager = StreamEmbeddingManager()
rg_manager = StreamRAGManager()


@app.get("/")
def root():
    return {"message": "Welcome to the Stream Embeddings API!"}


@app.post("/process/stream/{stream_id}")
def process_single_stream(stream_id: int):
    """Process a single stream and store its embedding."""
    try:
        eb_manager.process_stream_for_embedding(stream_id)
        return {"message": f"Stream {stream_id} processed successfully."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/all")
def process_all_streams():
    """Process all streams."""
    try:
        eb_manager.process_all_streams()
        return {"message": "All streams processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_streams(request: QueryRequest):
    """Find the most similar streams to a user query."""
    try:
        results = eb_manager.find_closest_streams(request.user_query, request.top_n)
        return {"streams": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/explain_query_stream")
def explain_query_stream(request: ExplainRequest):
    """Explain the relationship between user's query and each data stream"""
    try:
        results = rg_manager.explain_relationship(user_query=request.user_query, streams=request.streams)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))