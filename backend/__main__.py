import argparse

import uvicorn

parser = argparse.ArgumentParser(description="Start the local OSNet API and HTML page")
parser.add_argument("--host", default="0.0.0.0",
                    help="Bind address (default: 0.0.0.0; use 127.0.0.1 for local-only access)")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
uvicorn.run("backend.app:app", host=args.host, port=args.port)
