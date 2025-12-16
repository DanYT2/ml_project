#!/usr/bin/env python3
"""
Run the ASL Gesture Recognition API server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000
    python scripts/run_api.py --reload  # For development
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run the ASL API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    import uvicorn
    
    print("=" * 60)
    print("ASL Gesture Recognition API")
    print("=" * 60)
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API docs available at http://{args.host}:{args.port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()












