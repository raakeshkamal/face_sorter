#!/bin/bash

# face-sorter development starter script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Face Sorter Development Environment...${NC}"

# 1. Check for required tools
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ 'uv' is not installed.${NC} Please install it first: https://github.com/astral-sh/uv"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ 'npm' is not installed.${NC} Please install Node.js and npm."
    exit 1
fi

# 2. Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found! Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit the .env file to set your SOURCE_DIR and other paths.${NC}"
fi

# 3. Check for MongoDB (Standard port 27017)
if ! nc -z localhost 27017 2>/dev/null; then
    echo -e "${RED}❌ MongoDB is not running on localhost:27017.${NC}"
    echo "Please start MongoDB before running this script."
    exit 1
fi

# Function to handle cleanup on exit
cleanup() {
    echo -e "\n${BLUE}🛑 Shutting down services...${NC}"
    # Kill the entire process group to ensure sub-processes are cleaned up
    kill -- -$$ 2>/dev/null
    exit
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM EXIT

# 4. Start Backend
echo -e "${GREEN}📡 Starting Backend Server...${NC}"
# Using 'uv run' to ensure we use the project's virtual environment
uv run face-sorter web &
BACKEND_PID=$!

# 5. Start Frontend
echo -e "${GREEN}💻 Starting Frontend Development Server...${NC}"
cd src/face_sorter/web/frontend || exit 1

# Install dependencies if node_modules is missing
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
    npm install --no-audit --no-fund
fi

npm run dev &
FRONTEND_PID=$!

# Go back to root for convenience if needed, though we'll just wait here
cd - > /dev/null

echo -e "${GREEN}✅ Both services are starting!${NC}"
echo -e "${BLUE}Backend URL: ${NC}http://127.0.0.1:8000"
echo -e "${BLUE}Frontend URL: ${NC}http://127.0.0.1:5173 (Access this one for development)"
echo -e "${YELLOW}Press Ctrl+C to stop both servers.${NC}"

# Wait for background processes
wait
