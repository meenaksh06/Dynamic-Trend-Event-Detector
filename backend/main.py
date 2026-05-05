from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import asyncio
import json

# Ensure the backend directory is in the path for direct execution
sys.path.insert(0, os.path.dirname(__file__))
from service import AnalyticsService

app = FastAPI(title="Dynamic Trend & Event Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "processed_featured_data.csv")
service = AnalyticsService(DATA_PATH)


# ══════════════════════════════════════════════════════
# WebSocket Connection Manager
# ══════════════════════════════════════════════════════

class ConnectionManager:
    """Manages active WebSocket connections for live data broadcast."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"  ⚡ WS connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"  ⚡ WS disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        """Send data to all connected WebSocket clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# ══════════════════════════════════════════════════════
# Background Live Event Broadcaster
# ══════════════════════════════════════════════════════

async def live_event_broadcaster():
    """Background task: generate and broadcast a simulated live event every 3 seconds."""
    while True:
        await asyncio.sleep(3)
        if manager.active_connections:
            try:
                event = service.generate_live_event()
                await manager.broadcast(event)
            except Exception as e:
                print(f"  Broadcast error: {e}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(live_event_broadcaster())
    print("  ⚡ Live event broadcaster started (every 3s)\n")


# ══════════════════════════════════════════════════════
# WebSocket Endpoint
# ══════════════════════════════════════════════════════

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive — wait for client messages (ping/pong)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ══════════════════════════════════════════════════════
# REST API Endpoints
# ══════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"message": "Dynamic Trend & Event Detector API", "status": "active"}


@app.get("/api/stats")
async def get_stats():
    return service.get_stats()


@app.get("/api/topics")
async def get_topics():
    return service.get_topics()


@app.get("/api/trends")
async def get_trends():
    return service.get_trends()


@app.get("/api/sentiment-timeline")
async def get_sentiment_timeline():
    return service.get_sentiment_timeline()


@app.get("/api/forecast/{topic_id}")
async def get_forecast(topic_id: int):
    forecast = service.get_forecast(topic_id)
    if not forecast:
        raise HTTPException(status_code=404, detail="Forecast unavailable")
    return forecast


@app.get("/api/hybrid-forecast/{topic_id}")
async def get_hybrid_forecast(topic_id: int):
    result = service.get_hybrid_forecast(topic_id)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/ablation")
async def get_ablation():
    return service.get_ablation()


@app.get("/api/articles/{topic_id}")
async def get_articles(topic_id: int, limit: int = 10):
    return service.get_articles(topic_id, limit)


@app.get("/api/live-stats")
async def get_live_stats():
    return service.get_live_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
