from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
from .service import AnalyticsService

app = FastAPI(title="Dynamic Trend & Event Detector API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/processed_featured_data.csv")
service = AnalyticsService(DATA_PATH)

@app.get("/")
async def root():
    return {"message": "Welcome to the Dynamic Trend & Event Detector API", "status": "active"}

@app.get("/api/stats")
async def get_stats():
    return service.get_stats()

@app.get("/api/topics")
async def get_topics():
    return service.get_topics()

@app.get("/api/trends")
async def get_trends():
    return service.get_trends()

@app.get("/api/forecast/{topic_id}")
async def get_forecast(topic_id: int):
    forecast = service.get_forecast(topic_id)
    if not forecast:
        raise HTTPException(status_code=404, detail="Forecast not available or topic_id invalid")
    return forecast

@app.get("/api/articles/{topic_id}")
async def get_articles(topic_id: int, limit: int = 10):
    articles = service.get_articles(topic_id, limit)
    return articles

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
