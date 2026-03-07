#!/bin/bash

# face-sorter development starter script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Face Sorter Development Environment...${NC}"

# 1. Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found! Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit the .env file to set your SOURCE_DIR and other paths.${NC}"
fi

# 2. Check for MongoDB (Standard port 27017)
if ! nc -z localhost 27017 2>/dev/null; then
    echo -e "${RED}❌ MongoDB is not running on localhost:27017.${NC}"
    echo "Please start MongoDB before running this script."
    exit 1
fi

# Function to handle cleanup on exit
cleanup() {
    echo -e "\n${BLUE}🛑 Shutting down services...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM EXIT

# 3. Start Backend
echo -e "${GREEN}📡 Starting Backend Server...${NC}"
# Using 'uv run' to ensure we use the project's virtual environment
uv run face-sorter web &
BACKEND_PID=$!

# 4. Start Frontend
echo -e "${GREEN}💻 Starting Frontend Development Server...${NC}"
cd src/face_sorter/web/frontend

# Install dependencies if node_modules is missing
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}📦 Installing frontend dependencies (first time)...${NC}"
    npm install
fi

npm run dev &
FRONTEND_PID=$!

echo -e "${GREEN}✅ Both services are starting!${NC}"
echo -e "${BLUE}Backend URL: ${NC}http://127.0.0.1:8000"
echo -e "${BLUE}Frontend URL: ${NC}http://127.0.0.1:5173 (Access this one for development)"
echo -e "${YELLOW}Press Ctrl+C to stop both servers.${NC}"

# Wait for background processes
wait
