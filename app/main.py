"""
FastAPI frontend for the Uber_Support agent.

Two pages:
  /       -- dashboard: renders eval/dashboard_data.json (run
             `python -m eval.generate_dashboard_data` first, after the
             eval harness, to refresh it)
  /demo   -- live demo: file a "case" (type a message), watch the agent
             classify / retrieve / draft / decide in real time via
             POST /api/file-case

Run with: uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
"""
import json
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.llm_client import get_llm_client
from src.retrieval import TfidfRetriever
from src.pipeline import run_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Uber_Support Agent Case File")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

_llm = None
_retriever = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm_client()
    return _llm


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = TfidfRetriever(str(BASE_DIR / "data" / "processed" / "message_units.csv"))
    return _retriever


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    data_path = BASE_DIR / "eval" / "dashboard_data.json"
    if data_path.exists():
        with open(data_path) as f:
            data = json.load(f)
        missing = False
    else:
        data = {}
        missing = True
    return templates.TemplateResponse(request, "dashboard.html", {
        "data": data, "missing": missing,
        "backend": type(get_llm()).__name__,
    })


@app.get("/demo", response_class=HTMLResponse)
def demo(request: Request):
    return templates.TemplateResponse(request, "demo.html", {
        "result": None, "message": "",
        "backend": type(get_llm()).__name__,
    })


@app.post("/demo", response_class=HTMLResponse)
def file_case(request: Request, message: str = Form(...)):
    result = None
    if message.strip():
        r = run_pipeline(message, get_llm(), get_retriever())
        result = r.to_dict()
    return templates.TemplateResponse(request, "demo.html", {
        "result": result, "message": message,
        "backend": type(get_llm()).__name__,
    })
