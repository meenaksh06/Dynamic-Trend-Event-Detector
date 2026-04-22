#!/bin/bash

# Configuration
export PATH=$PATH:/opt/homebrew/bin:/usr/local/bin
BACKEND_PORT=8000
FRONTEND_PORT=3000

echo "🚀 Starting Dynamic Trend & Event Detector Platform..."

# Start Backend
echo "📦 Starting FastAPI Backend on port $BACKEND_PORT..."
cd backend
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!

# Start Frontend
echo "🌐 Starting Next.js Frontend on port $FRONTEND_PORT..."
cd ../frontend
npm install
npm run dev -- -p $FRONTEND_PORT &
FRONTEND_PID=$!

# Handle shutdown
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM EXIT

echo "✅ Both services are running!"
echo "   - Backend: http://localhost:$BACKEND_PORT"
echo "   - Frontend: http://localhost:$FRONTEND_PORT"

# Keep script running
wait
