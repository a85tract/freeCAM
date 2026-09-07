"""The HTTP layer of the local service, kept apart so FastAPI is optional.

Written without deferred annotations on purpose: FastAPI reads the route
parameters' types, and a ``Request`` it cannot resolve becomes a query
parameter.
"""

from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .codegen import STATIC
from .document import WorkflowEditError
from .service import VERSION, ServiceRefused, WorkflowService


def build_app(service: WorkflowService, *, static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="freeCAM Workflow Builder", version=VERSION, docs_url=None, redoc_url=None)
    static = static_dir or STATIC

    def authorised(request: Request) -> None:
        token = request.headers.get("x-freecam-token")
        if token != service.token:
            raise HTTPException(status_code=401, detail="missing or wrong session token")
        origin = request.headers.get("origin")
        host = request.headers.get("host")
        if origin and host and urlsplit(origin).netloc != host:
            raise HTTPException(status_code=403, detail="cross-origin requests are refused")

    def guarded(call: Callable[[], Any]) -> Any:
        try:
            return call()
        except ServiceRefused as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except WorkflowEditError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/state")
    def state(request: Request) -> Any:
        authorised(request)
        return JSONResponse(service.state_payload())

    @app.put("/api/draft")
    def draft(request: Request, body: dict = Body(...)) -> Any:
        authorised(request)
        return guarded(lambda: service.save_draft(body["document"]))

    @app.post("/api/validate")
    def validate(request: Request, body: dict = Body(...)) -> Any:
        authorised(request)
        return guarded(lambda: service.validate(body["document"]))

    @app.post("/api/generate")
    def generate(request: Request, body: dict = Body(...)) -> Any:
        authorised(request)
        return guarded(lambda: service.generate(body["document"], dict(body.get("artifacts", {}))))

    @app.post("/api/run")
    def run(request: Request, body: dict = Body(...)) -> Any:
        authorised(request)
        return guarded(lambda: service.start_run(body["document"], int(body.get("steps", 1)),
                                                 bool(body.get("confirm_resources", False))))

    @app.get("/api/run")
    def run_status(request: Request) -> Any:
        authorised(request)
        return service.run_payload()

    @app.post("/api/stop")
    def stop(request: Request) -> Any:
        authorised(request)
        return guarded(service.stop)

    @app.post("/api/close")
    def close(request: Request) -> Any:
        authorised(request)
        return guarded(service.close_model)

    @app.get("/api/events")
    def events(request: Request, since: int = 0) -> Any:
        authorised(request)
        return {"events": service.events(since), "run": service.run_payload()}

    index = static / "index.html"
    if index.is_file():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/")
        def root() -> Any:
            return FileResponse(index)

        @app.get("/catalog.json")
        def catalog() -> Any:
            return JSONResponse(service.snapshot)
    else:
        @app.get("/")
        def missing() -> Any:
            return JSONResponse(
                {"detail": "the page is not built: run `npm install && npm run build` under web/ of the checkout"},
                status_code=503,
            )

    return app


__all__ = ["build_app"]
