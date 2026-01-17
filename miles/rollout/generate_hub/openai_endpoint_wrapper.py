from argparse import Namespace

import requests


class OpenAIEndpointTracer:
    def __init__(self, args: Namespace):
        router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        response = requests.post(f"{router_url}/sessions")
        response.raise_for_status()
        self.session_id = response.json()["session_id"]
        self.base_url = f"{router_url}/sessions/{self.session_id}"
        self.router_url = router_url

    def collect(self):
        response = requests.delete(f"{self.router_url}/sessions/{self.session_id}")
        response.raise_for_status()
        return response.json()["records"]
