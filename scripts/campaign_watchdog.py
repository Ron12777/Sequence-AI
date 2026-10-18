"""Stop only this Vast instance at a deadline, preserving its result files on disk."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import urllib.request


def instance_credentials(expected):
    env = dict(os.environ)
    # SSH shells may not inherit the container's startup environment.
    if not env.get("CONTAINER_API_KEY"):
        for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
            key, _, value = item.partition(b"=")
            if key in (b"CONTAINER_ID", b"CONTAINER_API_KEY"):
                env[key.decode()] = value.decode()
    if env.get("CONTAINER_ID") != str(expected) or not env.get("CONTAINER_API_KEY"):
        raise RuntimeError("Expected instance-scoped credentials are unavailable")
    return env["CONTAINER_API_KEY"]


def request(instance_id, key, stop=False):
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{instance_id}/",
        data=json.dumps({"state": "stopped"}).encode() if stop else None,
        method="PUT" if stop else "GET",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.load(response)
    if stop and result.get("success") is False:
        raise RuntimeError("Instance stop request was rejected")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", type=int, required=True)
    p.add_argument("--stop-at", required=True)
    p.add_argument("--probe", action="store_true")
    a = p.parse_args()
    key = instance_credentials(a.instance)
    request(a.instance, key)
    print("Instance-scoped API access verified", flush=True)
    if a.probe:
        return
    deadline = datetime.fromisoformat(a.stop_at.replace("Z", "+00:00")).timestamp()
    while time.time() < deadline:
        time.sleep(max(0, min(30, deadline - time.time())))
    for attempt in range(10):
        try:
            request(a.instance, key, stop=True)
            print("GPU stop requested; result disk retained", flush=True)
            return
        except Exception as exc:
            print(
                f"Stop attempt {attempt + 1} failed: {type(exc).__name__}", flush=True
            )
            time.sleep(30)
    raise RuntimeError("Unable to stop instance after retries")


if __name__ == "__main__":
    main()
