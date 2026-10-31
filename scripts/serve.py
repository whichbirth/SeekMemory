#!/usr/bin/env python3
"""Minimal memory kernel loopback service (host_contract = 1).

把内核**已有**的服务装配起来，用 HTTP 暴露 issue #107 P3 约定的七个端点。
不重写任何记忆逻辑：capture 走 Buffer.add（内部写 Source + 入队抽取），
recall 走 RecallService.preview，persona/status 走 StatusService。

标准库 only，不引入新依赖。

    python scripts/serve.py --data-dir <dir> [--port 0] [--token <t>] [--config cfg.json]

启动后写出 <data-dir>/runtime.json：{"port": ..., "token": ..., "host_contract": 1}

抽取值是 pull-style 的：内核从不主动调用模型。宿主调 /extract/plan 拿任务、
自己调模型、再 /extract/commit 回传——所以这里 provider_for 永远不该被调用。
"""

from __future__ import annotations

import argparse
import hmac
import json
import secrets
import sys
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.budget import ContextBudget  # noqa: E402
from core.domain.memory.graph import Graph  # noqa: E402
from perceive.buffer import Buffer  # noqa: E402
from recall.retriever import HybridRetriever  # noqa: E402
from services.ingest_service import IngestService  # noqa: E402
from services.recall_service import RecallService  # noqa: E402
from services.status_service import StatusService  # noqa: E402
from services.tree_service import TreeService  # noqa: E402
from store import open_store  # noqa: E402
from store.cache import Cache  # noqa: E402

HOST_CONTRACT = 1


def jsonable(value: Any) -> Any:
    """Project enums and dataclasses into JSON-compatible structures."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


class _Logger:
    """ServiceLogger 的最小实现：只输出摘要，不含正文。"""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def info(self, message: str) -> None:
        if not self.quiet:
            print(f"[serve] {message}", flush=True)


async def _no_model(_session: str | None = None):
    """provider_for 的占位：pull-style 抽取下内核永不调用模型。"""
    raise RuntimeError("serve.py never calls a model; extraction is pull-style")


class MemoryService:
    """Own the kernel services and map protocol requests onto them."""

    def __init__(self, data_dir: Path, config: dict[str, Any] | None = None,
                 quiet: bool = False) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.config = dict(config or {})
        self.logger = _Logger(quiet)

        self.store = open_store(str(data_dir / "memory.db"))
        self.cache = Cache(self.store.artifacts)
        self.cache.load()
        self.graph = Graph(self.store.artifacts)
        self.tree = TreeService(self.store.artifacts, self.cache)
        self.retriever = HybridRetriever(self.graph, self.cache)
        self.buffer = Buffer(str(data_dir / "pending.json"),
                             extraction=self.store.extraction)
        self.budget = ContextBudget(
            injection_chars=int(self.config.get("budget_chars", 600)),
        )

        def cfg(key: str, default: Any = None) -> Any:
            return self.config.get(key, default)

        self.owner = str(self.config.get("owner_umo", "loopback:default"))
        self.recall_service = RecallService(
            cache=self.cache,
            graph=self.graph,
            artifacts=self.store.artifacts,
            sources=self.store.sources,
            budget=self.budget,
            config=cfg,
            owner_umo=lambda: self.owner,
            hybrid_retriever=self.retriever,
            tree=self.tree,
            tree_enabled=bool(self.config.get("tree_enabled", False)),
            log=self.logger.info,
        )
        self.ingest = IngestService(
            buffer=self.buffer,
            artifacts=self.store.artifacts,
            sources=self.store.sources,
            extraction=self.store.extraction,
            cache=self.cache,
            graph=self.graph,
            journal=None,
            config=cfg,
            provider_for=_no_model,
            owner_id=lambda: self.owner,
            logger=self.logger,
        )
        self.status_service = StatusService(
            artifacts=self.store.artifacts, ledger=self.store.ledger,
        )
        # Tasks handed to the host but not yet committed; process-local on purpose.
        # 已交给宿主但尚未提交的任务；按设计仅保留在当前进程。
        self.tasks: dict[str, Any] = {}

    # ---------- 端点 ----------
    def healthz(self) -> tuple[int, dict]:
        return 200, {
            "ok": True,
            "version": "0.1.0",
            "host_contract": HOST_CONTRACT,
            "db": str(self.data_dir / "memory.db"),
            "uptime_s": round(time.time() - STARTED_AT),
        }

    def capture(self, body: dict) -> tuple[int, dict]:
        text = str(body.get("text") or "").strip()
        if not text:
            return 400, {"error": "text is required"}
        turn = self.buffer.add(
            str(body.get("subject_key") or self.owner),
            text,
            str(body.get("speaker") or body.get("sender") or ""),
            str(body.get("msg_id") or body.get("source_id") or ""),
            str(body.get("at") or ""),
            role=str(body.get("role") or "user"),
        )
        if turn is None:
            return 400, {"error": "rejected"}
        return 200, {"accepted": True, "source_id": turn.source_id}

    def recall(self, body: dict) -> tuple[int, dict]:
        subject = str(body.get("subject_key") or self.owner)
        query = str(body.get("query") or "")
        result = self.recall_service.preview(
            query,
            subject,
            bool(body.get("private", True)),
            str(body.get("speaker") or ""),
        )
        block = str(result.get("block") or "")
        visible = list(result.get("visible") or [])
        return 200, {
            "block": block,
            "chars": len(block),
            "items": [{"id": item_id} for item_id in visible],
            "info": jsonable(result.get("info") or {}),
        }

    def persona(self, query: dict) -> tuple[int, dict]:
        row = self.status_service.persona() or {}
        text = str(row.get("text") or "")
        return 200, {
            "text": text,
            "chars": len(text),
            "revision": int(row.get("revision") or 0),
        }

    def status(self) -> tuple[int, dict]:
        return 200, {
            "host_contract": HOST_CONTRACT,
            "phase": "idle",
            "progress": {"phase": "idle", "pending": len(self.buffer.items), "running": 0},
            "ledger_valid": bool(self.status_service.ledger_valid()),
            "uptime_s": round(time.time() - STARTED_AT),
        }

    def extract_plan(self, query: dict) -> tuple[int, dict]:
        max_turns = int(query.get("max_turns") or self.config.get("extract_task_turns") or 12)
        task = self.ingest.next_extraction_task(max_turns=max_turns)
        if task is None:
            return 204, {}
        self.tasks[task.task_id] = task
        return 200, {
            "task_id": task.task_id,
            "source_ids": list(task.source_ids),
            "session_id": task.session_id,
            "topic_id": task.topic_id,
            "turns": task.turns,
            "chat_request": {
                "system_prompt": task.system_prompt,
                "prompt": task.prompt,
                "max_tokens": int(self.config.get("extract_max_tokens", 800)),
                "temperature": 0,
            },
        }

    def extract_commit(self, body: dict) -> tuple[int, dict]:
        """Persist host-supplied model output through the kernel plan pipeline.

        宿主自己调模型，这里只负责校验并持久化；未知或已消费的任务返回 409。
        """
        task_id = str(body.get("task_id") or "")
        task = self.tasks.get(task_id)
        source_ids = [
            str(value) for value in (body.get("source_ids") or (task.source_ids if task else ()))
            if value
        ]
        if not source_ids:
            return 409, {"error": "task_stale", "detail": "unknown task_id; call /extract/plan again"}
        reply = body.get("chat_response")
        usage_body = body.get("usage") or {}
        if isinstance(reply, dict):
            raw = str(reply.get("text") or "")
            usage = (
                int(reply.get("usage_in") or 0),
                int(reply.get("usage_cached") or 0),
                int(reply.get("usage_out") or 0),
            )
        elif isinstance(reply, str):
            raw = reply
            usage = (
                int(usage_body.get("in") or 0),
                int(usage_body.get("cached") or 0),
                int(usage_body.get("out") or 0),
            )
        else:
            raw = ""
            usage = (0, 0, 0)
        if not raw.strip():
            return 400, {"error": "chat_response.text is required"}
        try:
            outcome = self.ingest.commit_extraction(source_ids, raw, usage=usage)
        except Exception as exc:  # 解析失败已在队列里标记重试，这里只回传结果
            return 422, {"error": "invalid_extraction", "detail": type(exc).__name__,
                         "retryable": True}
        if "task_stale" in outcome.errors:
            return 409, {"error": "task_stale", "retryable": False}
        self.tasks.pop(task_id, None)
        return 200, {
            "written": outcome.written,
            "skipped": outcome.skipped,
            "ids": list(outcome.ids),
            "errors": list(outcome.errors),
            "retryable": bool(outcome.retryable),
        }


STARTED_AT = time.time()


def make_handler(service: MemoryService, token: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # 静音默认日志
            return

        def _send(self, code: int, payload: dict | None) -> None:
            if code == 204 or payload is None:
                self.send_response(204)
                self.send_header("content-length", "0")
                self.end_headers()
                return
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _auth_ok(self) -> bool:
            header = self.headers.get("authorization") or ""
            return hmac.compare_digest(header, f"Bearer {token}")

        def _route(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                code, payload = service.healthz()
                return self._send(code, payload)
            if not self._auth_ok():
                return self._send(401, {"error": "unauthorized"})
            query = {}
            if "?" in self.path:
                from urllib.parse import parse_qs
                query = {k: v[-1] for k, v in parse_qs(self.path.split("?", 1)[1]).items()}
            try:
                if path == "/capture" and method == "POST":
                    return self._send(*service.capture(self._read_body()))
                if path == "/recall" and method == "POST":
                    return self._send(*service.recall(self._read_body()))
                if path == "/persona" and method == "GET":
                    return self._send(*service.persona(query))
                if path == "/status" and method == "GET":
                    return self._send(*service.status())
                if path == "/extract/plan" and method == "GET":
                    return self._send(*service.extract_plan(query))
                if path == "/extract/commit" and method == "POST":
                    return self._send(*service.extract_commit(self._read_body()))
            except Exception as exc:  # 兜底：不因单请求崩溃整服务
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return self._send(404, {"error": "not found"})

        def _read_body(self) -> dict:
            length = int(self.headers.get("content-length") or 0)
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

        def do_GET(self) -> None:  # noqa: N802
            self._route("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._route("POST")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="SeekMemory loopback kernel")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--token", default="")
    parser.add_argument("--config", default="", help="JSON file with kernel config")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    config = {}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    service = MemoryService(data_dir, config, quiet=args.quiet)
    token = args.token or secrets.token_urlsafe(24)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(service, token))
    port = server.server_address[1]

    runtime = {"port": port, "token": token, "host_contract": HOST_CONTRACT}
    (data_dir / "runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False), encoding="utf-8",
    )
    print(f"memory kernel listening on 127.0.0.1:{port}", flush=True)
    print(f"runtime file: {data_dir / 'runtime.json'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
