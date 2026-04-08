from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.eval.runner import run_eval
from src.rag.pipeline import RAGPipeline


class QueryRequest(BaseModel):
    query: str


class EvalRequest(BaseModel):
    dataset_path: str


def create_app(config_path: str) -> FastAPI:
    app = FastAPI(title="RAG Tool API")
    pipeline = RAGPipeline(config_path=config_path)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/query")
    def query(req: QueryRequest) -> dict:
        return pipeline.query(req.query)

    @app.post("/eval/run")
    def eval_run(req: EvalRequest) -> dict:
        return run_eval(config_path=config_path, dataset_path=req.dataset_path)

    return app
