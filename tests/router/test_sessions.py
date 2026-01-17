import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from miles.router.sessions import SessionManager, SessionRecord, setup_session_routes


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
        with pytest.raises(KeyError):
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
        with pytest.raises(KeyError):
            manager.add_record("nonexistent", record)


class TestSessionRoutes:
    def test_create_session(self, client):
        response = client.post("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session(self, client):
        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        get_resp = client.get(f"/sessions/{session_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, client):
        response = client.get("/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_delete_session(self, client):
        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        delete_resp = client.delete(f"/sessions/{session_id}")
        assert delete_resp.status_code == 200
        data = delete_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

        get_resp = client.get(f"/sessions/{session_id}")
        assert get_resp.status_code == 404

    def test_delete_session_not_found(self, client):
        response = client.delete("/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"


class TestSessionProxy:
    def test_proxy_json_request_response(self, app_with_sessions):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        mock_router._do_proxy.return_value = {
            "request_body": json.dumps({"prompt": "hello"}).encode(),
            "response_body": json.dumps({"result": "ok"}).encode(),
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }
        mock_router._build_response.return_value = JSONResponse(
            content={"result": "ok"}, status_code=200
        )

        proxy_resp = client.post(
            f"/sessions/{session_id}/generate",
            json={"prompt": "hello"},
        )

        assert proxy_resp.status_code == 200
        assert proxy_resp.json() == {"result": "ok"}

        mock_router._do_proxy.assert_called()

        get_resp = client.get(f"/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert len(records) == 1
        assert records[0]["method"] == "POST"
        assert records[0]["path"] == "generate"
        assert records[0]["request_json"] == {"prompt": "hello"}
        assert records[0]["response_json"] == {"result": "ok"}
        assert records[0]["status_code"] == 200

    def test_proxy_empty_request_body(self, app_with_sessions):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        mock_router._do_proxy.return_value = {
            "request_body": b"{}",
            "response_body": json.dumps({"status": "ok"}).encode(),
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }
        mock_router._build_response.return_value = JSONResponse(
            content={"status": "ok"}, status_code=200
        )

        proxy_resp = client.get(f"/sessions/{session_id}/health")

        assert proxy_resp.status_code == 200

        get_resp = client.get(f"/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert records[0]["request_json"] == {}
        assert records[0]["response_json"] == {"status": "ok"}

    def test_proxy_session_not_found(self, client):
        response = client.post("/sessions/nonexistent/generate", json={})
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_proxy_multiple_requests(self, app_with_sessions):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        for i in range(3):
            mock_router._do_proxy.return_value = {
                "request_body": json.dumps({"req": i}).encode(),
                "response_body": json.dumps({"i": i}).encode(),
                "status_code": 200,
                "headers": {},
            }
            mock_router._build_response.return_value = JSONResponse(
                content={"i": i}, status_code=200
            )

            client.post(f"/sessions/{session_id}/test", json={"req": i})

        get_resp = client.get(f"/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert len(records) == 3
        for i, record in enumerate(records):
            assert record["request_json"] == {"req": i}
            assert record["response_json"] == {"i": i}

    def test_proxy_different_http_methods(self, app_with_sessions):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        for method in methods:
            mock_router._do_proxy.return_value = {
                "request_body": b"{}",
                "response_body": json.dumps({"method": method}).encode(),
                "status_code": 200,
                "headers": {},
            }
            mock_router._build_response.return_value = JSONResponse(
                content={"method": method}, status_code=200
            )

            resp = client.request(method, f"/sessions/{session_id}/test")
            assert resp.status_code == 200

        get_resp = client.get(f"/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert len(records) == len(methods)
        for i, record in enumerate(records):
            assert record["method"] == methods[i]
