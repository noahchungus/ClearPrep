"""Start the app:  python run.py   (respects the PORT environment variable, as free hosts set it)."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("web.app:app", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")), reload=bool(os.environ.get("RELOAD")))
