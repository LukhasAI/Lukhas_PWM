#!/bin/bash
# Docker build helper script for LUKHAS AGI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Default values
TAG="latest"
PUSH=false
BUILD_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--tag)
      TAG="$2"
      shift 2
      ;;
    -p|--push)
      PUSH=true
      shift
      ;;
    --no-cache)
      BUILD_ARGS="$BUILD_ARGS --no-cache"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -t, --tag TAG    Docker image tag (default: latest)"
      echo "  -p, --push       Push image to registry after build"
      echo "  --no-cache       Build without using cache"
      echo "  -h, --help       Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

echo -e "${GREEN}Building LUKHAS AGI Docker image...${NC}"
echo "Tag: lukhas/lukhas-agi:$TAG"

# Build the image
docker build $BUILD_ARGS -t lukhas/lukhas-agi:$TAG .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
    
    # Tag as latest if not already
    if [ "$TAG" != "latest" ]; then
        docker tag lukhas/lukhas-agi:$TAG lukhas/lukhas-agi:latest
        echo -e "${GREEN}Tagged as latest${NC}"
    fi
    
    # Push if requested
    if [ "$PUSH" = true ]; then
        echo -e "${YELLOW}Pushing to registry...${NC}"
        docker push lukhas/lukhas-agi:$TAG
        if [ "$TAG" != "latest" ]; then
            docker push lukhas/lukhas-agi:latest
        fi
        echo -e "${GREEN}Push complete!${NC}"
    fi
    
    # Show image info
    echo -e "\n${GREEN}Image info:${NC}"
    docker images lukhas/lukhas-agi:$TAG
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi