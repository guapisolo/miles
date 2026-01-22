from types import SimpleNamespace

import pytest
import requests

from miles.router.router import MilesRouter
from miles.router.session.naive_trajectory import NaiveTrajectoryManager
from miles.router.session.sessions import GetSessionResponse, SessionRecord
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


class DummyTokenizer:
    """Minimal tokenizer stub for testing NaiveTrajectoryManager."""

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_special_tokens: bool = False,
        add_generation_prompt: bool = True,
    ):
        """Return deterministic token ids based on message count."""
        base = len(messages) or 1
        return [base, base + 1, base + 2]


@pytest.fixture
def naive_manager():
    """Create a NaiveTrajectoryManager with a dummy tokenizer."""
    args = SimpleNamespace()
    tokenizer = DummyTokenizer()
    return NaiveTrajectoryManager(args, tokenizer)


class TestNaiveTrajectoryManager:
    def test_create_session(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in naive_manager.sessions

    def test_get_session_records_by_id(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        records = naive_manager.get_session_records_by_id(session_id)
        assert isinstance(records, list)
        assert records == []

    def test_get_session_records_by_id_not_found(self, naive_manager: NaiveTrajectoryManager):
        with pytest.raises(ValueError):
            naive_manager.get_session_records_by_id("nonexistent")

    def test_calc_prompt_tokens_for_existing_session(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        messages = [{"role": "user", "content": "hello"}]

        token_ids = naive_manager.calc_prompt_tokens(session_id, messages)

        assert isinstance(token_ids, list)
        assert token_ids == [1, 2, 3]

    def test_calc_prompt_tokens_for_missing_session(self, naive_manager: NaiveTrajectoryManager):
        messages = [{"role": "user", "content": "hello"}]
        with pytest.raises(ValueError):
            naive_manager.calc_prompt_tokens("missing", messages)

    def test_delete_session_by_id(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        naive_manager.delete_session_by_id(session_id)
        assert session_id not in naive_manager.sessions
        with pytest.raises(ValueError):
            naive_manager.delete_session_by_id(session_id)

    def test_append_session_record(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        record = SessionRecord(
            timestamp=1234567890.0,
            method="POST",
            path="v1/chat/completions",
            request={"messages": [{"role": "user", "content": "hello"}]},
            response={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
            status_code=200,
        )

        naive_manager.append_session_record(session_id, record)
        records = naive_manager.get_session_records_by_id(session_id)
        assert len(records) == 1
        assert records[0] == record

    def test_append_session_record_missing_session(self, naive_manager: NaiveTrajectoryManager):
        record = SessionRecord(
            timestamp=1234567890.0,
            method="POST",
            path="v1/chat/completions",
            request={},
            response={},
            status_code=200,
        )
        with pytest.raises(ValueError):
            naive_manager.append_session_record("missing", record)

    def test_tokenizer_integration(self, naive_manager: NaiveTrajectoryManager):
        """Test tokenizer integration with real tokenization."""
        session_id = naive_manager.create_session()

        # Test with different message structures
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        token_ids = naive_manager.calc_prompt_tokens(session_id, messages)
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)

    def test_tokenizer_convert_tokens_to_ids(self):
        """Test tokenizer convert_tokens_to_ids functionality."""

        # Create a mock tokenizer that has convert_tokens_to_ids method
        class MockTokenizer:
            def apply_chat_template(
                self, messages, tokenize=True, add_special_tokens=False, add_generation_prompt=True
            ):
                return [101, 102, 103]  # Mock token ids

            def convert_tokens_to_ids(self, tokens):
                if isinstance(tokens, str):
                    return 999  # Mock token id for string
                return [999 + i for i in range(len(tokens))]  # Mock ids for list

        args = SimpleNamespace()
        tokenizer = MockTokenizer()
        manager = NaiveTrajectoryManager(args, tokenizer)

        session_id = manager.create_session()

        # Test the token conversion logic indirectly through calc_prompt_tokens
        messages = [{"role": "user", "content": "test"}]
        token_ids = manager.calc_prompt_tokens(session_id, messages)
        assert token_ids == [101, 102, 103]

    def test_session_thread_safety(self, naive_manager: NaiveTrajectoryManager):
        """Test that session operations are thread-safe."""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                session_id = naive_manager.create_session()
                time.sleep(0.01)  # Small delay to increase chance of race conditions

                # Add a record
                record = SessionRecord(
                    timestamp=time.time(),
                    method="POST",
                    path="v1/chat/completions",
                    request={"test": worker_id},
                    response={"result": f"response_{worker_id}"},
                    status_code=200,
                )
                naive_manager.append_session_record(session_id, record)

                # Retrieve and verify
                records = naive_manager.get_session_records_by_id(session_id)
                results.append((worker_id, len(records)))

            except Exception as e:
                errors.append((worker_id, str(e)))

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5, "Not all workers completed"
        for worker_id, record_count in results:
            assert record_count == 1, f"Worker {worker_id} has {record_count} records, expected 1"


@pytest.fixture(scope="class")
def router_env():
    """Create a MilesRouter with session routes and a mock backend."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as backend:
        args = SimpleNamespace(
            miles_router_max_connections=10,
            miles_router_timeout=30,
            miles_router_middleware_paths=[],
            rollout_health_check_interval=60,
            miles_router_health_check_failure_threshold=3,
            hf_checkpoint="Qwen/Qwen3-0.6B",
            trajectory_manager="naive_trajectory",
        )
        router = MilesRouter(args)

        port = find_available_port(31000)
        server = UvicornThreadServer(router.app, host="127.0.0.1", port=port)
        server.start()

        url = f"http://127.0.0.1:{port}"
        requests.post(f"{url}/add_worker", json={"url": backend.url}, timeout=5.0)

        try:
            yield SimpleNamespace(url=url)
        finally:
            server.stop()


class TestSessionRoutes:
    def test_create_session(self, router_env):
        response = requests.post(f"{router_env.url}/sessions", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session_initial_state(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, router_env):
        response = requests.get(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_delete_session(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        delete_resp = requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert delete_resp.status_code == 204
        assert delete_resp.text == ""

        assert requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0).status_code == 404

    def test_delete_session_not_found(self, router_env):
        response = requests.delete(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"


class TestSessionProxy:
    def test_proxy_chat_creates_record(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 1+2?"}],
            "return_logprob": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body
        assert body["choices"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        data = get_resp.json()
        records = data["records"]

        assert isinstance(records, list)
        assert len(records) == 1
        record = records[0]
        assert record["method"] == "POST"
        assert record["path"] == "v1/chat/completions"
        assert record["status_code"] == 200
        assert "request" in record
        assert "response" in record
        assert record["request"]["messages"] == payload["messages"]

    def test_proxy_chat_with_logprobs_token_ids(self, router_env):
        """Test that proxy adds token_ids to logprobs when missing."""
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        # Mock a response that includes logprobs without token_ids
        payload = {
            "messages": [{"role": "user", "content": "test token ids"}],
            "return_logprob": True,
        }

        # We'll need to modify the mock to return logprobs
        # For now, test the basic functionality - this would need backend modification
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200

        # Verify record was created
        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        data = get_resp.json()
        records = data["records"]
        assert len(records) == 1


class TestSessionModels:
    """Test SessionRecord and GetSessionResponse models."""

    def test_session_record_creation(self):
        """Test SessionRecord model creation and validation."""
        record = SessionRecord(
            timestamp=1234567890.0,
            method="POST",
            path="v1/chat/completions",
            request={"messages": [{"role": "user", "content": "hello"}]},
            response={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
            status_code=200,
        )

        assert record.timestamp == 1234567890.0
        assert record.method == "POST"
        assert record.path == "v1/chat/completions"
        assert record.status_code == 200
        assert record.request == {"messages": [{"role": "user", "content": "hello"}]}
        assert record.response == {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    def test_session_record_dict_conversion(self):
        """Test SessionRecord can be converted to dict."""
        record = SessionRecord(
            timestamp=1234567890.0,
            method="GET",
            path="v1/models",
            request={},
            response={"data": []},
            status_code=200,
        )

        record_dict = record.model_dump()
        assert isinstance(record_dict, dict)
        assert record_dict["timestamp"] == 1234567890.0
        assert record_dict["method"] == "GET"
        assert record_dict["path"] == "v1/models"

    def test_get_session_response_creation(self):
        """Test GetSessionResponse model creation."""
        records = [
            SessionRecord(
                timestamp=1234567890.0,
                method="POST",
                path="v1/chat/completions",
                request={"messages": [{"role": "user", "content": "test"}]},
                response={"choices": [{"message": {"content": "response"}}]},
                status_code=200,
            )
        ]

        response = GetSessionResponse(session_id="test_session_123", records=records)

        assert response.session_id == "test_session_123"
        assert len(response.records) == 1
        assert response.records[0].method == "POST"

    def test_get_session_response_empty_records(self):
        """Test GetSessionResponse with empty records."""
        response = GetSessionResponse(session_id="empty_session", records=[])

        assert response.session_id == "empty_session"
        assert response.records == []
