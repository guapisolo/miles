import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from miles.router.router import MilesRouter
from miles.router.sessions import SessionManager, SessionRecord, setup_session_routes
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server


@pytest.fixture
def mock_router():
    router = MagicMock()
    router._do_proxy = AsyncMock()
    router._build_response = MagicMock()
    return router


@pytest.fixture
def app_with_sessions(mock_router):
    app = FastAPI()
    setup_session_routes(app, mock_router)
    return app, mock_router


@pytest.fixture
def client(app_with_sessions):
    app, _ = app_with_sessions
    return TestClient(app)


class TestSessionManager:
    def test_create_session(self):
        manager = SessionManager()
        session_id = manager.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in manager.sessions
        assert manager.sessions[session_id] == []

    def test_get_session_exists(self):
        manager = SessionManager()
        session_id = manager.create_session()
        records = manager.get_session(session_id)
        assert records == []

    def test_get_session_not_exists(self):
        manager = SessionManager()
        records = manager.get_session("nonexistent")
        assert records is None

    def test_delete_session_exists(self):
        manager = SessionManager()
        session_id = manager.create_session()
        records = manager.delete_session(session_id)
        assert records == []
        assert session_id not in manager.sessions

    def test_delete_session_not_exists(self):
        manager = SessionManager()
        with pytest.raises(AssertionError):
            manager.delete_session("nonexistent")

    def test_add_record(self):
        manager = SessionManager()
        session_id = manager.create_session()
        record = SessionRecord(
            timestamp=1234567890.0,
            method="POST",
            path="generate",
            request_json={"prompt": "hello"},
            response_json={"text": "world"},
            status_code=200,
        )
        manager.add_record(session_id, record)
        assert len(manager.sessions[session_id]) == 1
        assert manager.sessions[session_id][0] == record

    def test_add_record_nonexistent_session(self):
        manager = SessionManager()
        record = SessionRecord(
            timestamp=1234567890.0,
            method="POST",
            path="generate",
            request_json={},
            response_json={},
            status_code=200,
        )
        with pytest.raises(AssertionError):
            manager.add_record("nonexistent", record)


class TestSessionRoutes:
    def test_create_session(self, client):
        response = client.post("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_delete_session(self, client):
        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        delete_resp = client.delete(f"/sessions/{session_id}")
        assert delete_resp.status_code == 200
        data = delete_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

        delete_again = client.delete(f"/sessions/{session_id}")
        assert delete_again.status_code == 404

    def test_delete_session_not_found(self, client):
        response = client.delete("/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"


class TestSessionProxy:
    def test_proxy_session_not_found(self, client):
        response = client.post("/sessions/nonexistent/generate", json={})
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE", "PATCH"])
    def test_proxy_records_request_response(self, app_with_sessions, method):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        session_id = client.post("/sessions").json()["session_id"]

        mock_router._do_proxy.return_value = {
            "request_body": json.dumps({"input": "data"}).encode(),
            "response_body": json.dumps({"output": "result"}).encode(),
            "status_code": 200,
            "headers": {},
        }
        mock_router._build_response.return_value = JSONResponse(content={"output": "result"}, status_code=200)

        resp = client.request(method, f"/sessions/{session_id}/test")
        assert resp.status_code == 200

        records = client.delete(f"/sessions/{session_id}").json()["records"]
        assert len(records) == 1
        assert records[0]["method"] == method
        assert records[0]["path"] == "test"
        assert records[0]["request_json"] == {"input": "data"}
        assert records[0]["response_json"] == {"output": "result"}

    def test_proxy_accumulates_records(self, app_with_sessions):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        session_id = client.post("/sessions").json()["session_id"]

        for i in range(3):
            mock_router._do_proxy.return_value = {
                "request_body": json.dumps({"i": i}).encode(),
                "response_body": json.dumps({"i": i}).encode(),
                "status_code": 200,
                "headers": {},
            }
            mock_router._build_response.return_value = JSONResponse(content={"i": i}, status_code=200)
            client.post(f"/sessions/{session_id}/test")

        records = client.delete(f"/sessions/{session_id}").json()["records"]
        assert len(records) == 3
        assert [r["request_json"]["i"] for r in records] == [0, 1, 2]


class TestSessionProxyIntegration:
    @pytest.fixture
    def real_router_client(self):
        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

        with with_mock_server(process_fn=process_fn) as server:
            args = SimpleNamespace(
                miles_router_max_connections=10,
                miles_router_timeout=30,
                miles_router_middleware_paths=[],
                rollout_health_check_interval=60,
                miles_router_health_check_failure_threshold=3,
            )
            router = MilesRouter(args)
            router.worker_request_counts[server.url] = 0
            router.worker_failure_counts[server.url] = 0
            yield TestClient(router.app), server

    def test_real_proxy_records_request_response(self, real_router_client):
        client, server = real_router_client

        session_id = client.post("/sessions").json()["session_id"]

        resp = client.post(
            f"/sessions/{session_id}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
        )
        assert resp.status_code == 200
        assert "text" in resp.json()

        records = client.delete(f"/sessions/{session_id}").json()["records"]
        assert len(records) == 1
        assert records[0]["method"] == "POST"
        assert records[0]["path"] == "generate"
        assert records[0]["request_json"]["input_ids"] == [1, 2, 3]
        assert "text" in records[0]["response_json"]
