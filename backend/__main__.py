import argparse

import uvicorn

parser = argparse.ArgumentParser(description="Start the local OSNet API and HTML page")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
uvicorn.run("backend.app:app", host="127.0.0.1", port=args.port)
