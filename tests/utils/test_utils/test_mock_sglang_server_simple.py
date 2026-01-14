#!/usr/bin/env python3
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

try:
    import httpx
except ImportError:
    print("httpx not available, skipping HTTP tests")
    httpx = None

from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, start_mock_server


def test_basic():
    print("Test 1: Basic server start/stop")
    server = MockSGLangServer(response_text="Test response", finish_reason="stop")
    try:
        server.start()
        print(f"  ✓ Server started on {server.url}")
        assert server.port > 0
        assert f"http://{server.host}:{server.port}" == server.url
        print("  ✓ Server URL is correct")
    finally:
        server.stop()
        print("  ✓ Server stopped")
    print()


def test_generate_endpoint():
    if httpx is None:
        print("Test 2: Generate endpoint (skipped - httpx not available)")
        return
    
    print("Test 2: Generate endpoint")
    server = MockSGLangServer(response_text="Hello, world!", finish_reason="stop", prompt_tokens=5, cached_tokens=2)
    try:
        server.start()
        time.sleep(0.5)  # Give server time to start
        
        response = httpx.post(
            f"{server.url}/generate",
            json={
                "input_ids": [1, 2, 3, 4, 5],
                "sampling_params": {"temperature": 0.7, "max_new_tokens": 10},
            },
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "text" in data
        assert data["text"] == "Hello, world!"
        assert "meta_info" in data
        assert data["meta_info"]["finish_reason"]["type"] == "stop"
        assert data["meta_info"]["prompt_tokens"] == 5
        assert data["meta_info"]["cached_tokens"] == 2
        print("  ✓ Response format is correct")
        
        assert len(server.requests) == 1
        assert server.requests[0]["input_ids"] == [1, 2, 3, 4, 5]
        print("  ✓ Request was recorded")
    finally:
        server.stop()
    print()


def test_finish_reasons():
    if httpx is None:
        print("Test 3: Finish reasons (skipped - httpx not available)")
        return
    
    print("Test 3: Finish reasons")
    for finish_reason in ["stop", "length", "abort"]:
        server = MockSGLangServer(response_text="Test", finish_reason=finish_reason, completion_tokens=32)
        try:
            server.start()
            time.sleep(0.5)
            
            response = httpx.post(f"{server.url}/generate", json={"input_ids": [], "sampling_params": {}}, timeout=5.0)
            assert response.status_code == 200
            data = response.json()
            
            assert data["meta_info"]["finish_reason"]["type"] == finish_reason
            if finish_reason == "length":
                assert "length" in data["meta_info"]["finish_reason"]
            print(f"  ✓ finish_reason='{finish_reason}' works correctly")
        finally:
            server.stop()
    print()


def test_return_logprob():
    if httpx is None:
        print("Test 4: Return logprob (skipped - httpx not available)")
        return
    
    print("Test 4: Return logprob")
    server = MockSGLangServer(response_text="Test", finish_reason="stop", completion_tokens=3)
    try:
        server.start()
        time.sleep(0.5)
        
        response = httpx.post(
            f"{server.url}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "output_token_logprobs" in data["meta_info"]
        logprobs = data["meta_info"]["output_token_logprobs"]
        assert isinstance(logprobs, list)
        assert len(logprobs) == 3
        assert isinstance(logprobs[0], list)
        assert len(logprobs[0]) == 2
        print("  ✓ output_token_logprobs format is correct")
    finally:
        server.stop()
    print()


def test_context_manager():
    if httpx is None:
        print("Test 5: Context manager (skipped - httpx not available)")
        return
    
    print("Test 5: Context manager")
    with start_mock_server(response_text="Context test", finish_reason="stop") as server:
        time.sleep(0.5)
        response = httpx.post(f"{server.url}/generate", json={"input_ids": [], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Context test"
        print("  ✓ Context manager works correctly")
    print()


if __name__ == "__main__":
    print("Running mock_sglang_server tests...\n")
    
    try:
        test_basic()
        test_generate_endpoint()
        test_finish_reasons()
        test_return_logprob()
        test_context_manager()
        
        print("All tests passed! ✓")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
