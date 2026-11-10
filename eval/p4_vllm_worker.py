"""vLLM executable selection and readiness checks."""

from __future__ import annotations

import argparse

import json

import os

import re

import signal

import subprocess

import time

import urllib.request

from pathlib import Path

import p4_eval_vllm as vclient

VLLM_PY = "vllm"

def wait_ready(port: int, timeout: int, proc=None) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vllm serve exited during startup with code {proc.returncode}; "
                "inspect the serve_<arm>-<step>_<port>.log file beside --output"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    # extra: models list non-empty
                    data = json.loads(resp.read())
                    if data.get("data"):
                        return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise TimeoutError(f"vllm serve on :{port} not ready within {timeout}s")
