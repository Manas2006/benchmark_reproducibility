#!/bin/bash

echo "🚀 Starting Qwen Math Evaluation UI..."

# Check if we're in the right directory
if [ ! -f "backend/app/main.py" ]; then
    echo "❌ Error: Please run this script from the qwen-eval-ui directory"
    exit 1
fi

# Start backend
echo "📡 Starting backend server on http://localhost:8000..."
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🌐 Starting frontend server on http://localhost:3002..."
cd frontend
python3 -m http.server 3002 &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Qwen Math Evaluation UI is running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3002"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for background processes
wait 