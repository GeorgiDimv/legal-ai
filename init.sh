#!/bin/bash
#
# Legal AI Document Processing Pipeline - Initialization Script
# Run this on your 6-GPU server to set up the complete pipeline
#

set -e

echo "=============================================="
echo "  Legal AI Pipeline - Initialization"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "[1/6] Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo -e "${RED}Docker Compose not found. Please install Docker Compose v2.${NC}"
    exit 1
fi

# Check NVIDIA Docker runtime
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected.${NC}"
    echo "Install with: sudo apt-get install -y nvidia-docker2"
fi

# Check GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo -e "  GPUs detected: ${GREEN}$GPU_COUNT${NC}"

if [ "$GPU_COUNT" -lt 6 ]; then
    echo -e "${YELLOW}Warning: Expected 6 GPUs, found $GPU_COUNT${NC}"
    echo "The LLM service requires 6 GPUs for tensor parallelism."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "  ${GREEN}Prerequisites OK${NC}"
echo ""

echo "[2/6] Setting up environment..."

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"

    # Prompt for HuggingFace token
    echo ""
    read -p "Enter your HuggingFace token (hf_xxxxx): " HF_TOKEN
    if [ -n "$HF_TOKEN" ]; then
        sed -i "s/HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
    fi

    # Generate random password for PostgreSQL
    POSTGRES_PASS=$(openssl rand -base64 16 | tr -d '/+=' | head -c 16)
    sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$POSTGRES_PASS/" .env
    echo "  Generated PostgreSQL password"

    echo -e "  ${GREEN}.env configured${NC}"
else
    echo "  .env already exists, skipping"
fi

echo ""

echo "[3/6] Creating HuggingFace cache directory..."
mkdir -p ~/.cache/huggingface
echo -e "  ${GREEN}Cache directory ready${NC}"
echo ""

echo "[4/6] Building Docker images..."
docker compose build --parallel
echo -e "  ${GREEN}Images built${NC}"
echo ""

echo "[5/6] Starting services..."
docker compose up -d
echo -e "  ${GREEN}Services started${NC}"
echo ""

echo "[6/6] Waiting for services to be ready..."
echo ""

# Function to check service health
check_service() {
    local name=$1
    local url=$2
    local max_attempts=$3
    local attempt=1

    printf "  %-20s " "$name"

    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}Ready${NC}"
            return 0
        fi
        printf "."
        sleep 5
        attempt=$((attempt + 1))
    done

    echo -e "${YELLOW}Not ready (may still be starting)${NC}"
    return 1
}

echo "Checking service health (this may take a few minutes)..."
echo ""

# PostgreSQL should be ready quickly
check_service "PostgreSQL" "localhost:5432" 12 || true

# Redis should be ready quickly
check_service "Redis" "localhost:6379" 6 || true

# OCR service
check_service "OCR Service" "http://localhost:8001/health" 24 || true

# Car Value service
check_service "Car Value" "http://localhost:8003/health" 12 || true

# Gateway
check_service "API Gateway" "http://localhost:80/health" 12 || true

# LLM takes longest (model download + loading)
echo ""
echo "  LLM Service (vLLM) - This may take 10-20 minutes on first run..."
echo "  (Downloading ~20GB model weights)"
echo ""
check_service "LLM Service" "http://localhost:8000/health" 240 || true

# Nominatim takes longest (OSM data import)
echo ""
echo "  Nominatim - OSM import takes 30-60 minutes on first run..."
echo "  You can check progress with: docker logs -f legal-ai-nominatim-1"
echo ""

echo "=============================================="
echo "  Initialization Complete!"
echo "=============================================="
echo ""
echo "Services:"
echo "  - API Gateway:    http://localhost:80"
echo "  - LLM (vLLM):     http://localhost:8000"
echo "  - OCR:            http://localhost:8001"
echo "  - Nominatim:      http://localhost:8002"
echo "  - Car Value:      http://localhost:8003"
echo "  - PostgreSQL:     localhost:5432"
echo "  - Redis:          localhost:6379"
echo ""
echo "Commands:"
echo "  docker compose logs -f          # View all logs"
echo "  docker compose logs -f llm      # View LLM logs (model loading)"
echo "  docker compose logs -f nominatim # View Nominatim logs (OSM import)"
echo "  docker compose ps               # Check container status"
echo "  watch -n 1 nvidia-smi           # Monitor GPU usage"
echo ""
echo "Test the pipeline:"
echo "  python3 test_pipeline.py --health-only  # Check service health"
echo "  python3 test_pipeline.py sample.pdf     # Process a document"
echo ""
