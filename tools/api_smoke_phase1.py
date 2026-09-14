"""Small dependency-free API smoke test for the Phase 1 System B contract."""

import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_json(url, method="GET", body=None, expected_status=(200,)):
    request = Request(url, method=method, data=body)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        response = urlopen(request, timeout=10)
        status = response.status
        payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        status = error.code
        payload = json.loads(error.read().decode("utf-8"))
    if status not in expected_status:
        raise AssertionError(f"unexpected status {status} for {url}")
    if not isinstance(payload, dict):
        raise AssertionError(f"response is not a JSON object for {url}")
    return payload


def main(base_url):
    base_url = base_url.rstrip("/")
    live = request_json(f"{base_url}/health/live")
    assert live.get("status") == "alive"

    ready = request_json(f"{base_url}/health/ready", expected_status=(200, 503))
    assert ready.get("status") in {"ready", "degraded"}
    assert isinstance(ready.get("checks"), dict)

    info = request_json(f"{base_url}/api/params/info")
    assert info and all(isinstance(value, dict) for value in info.values())

    invalid = request_json(
        f"{base_url}/api/params",
        method="POST",
        body=b'{"__unknown_phase1_param__": 1}',
        expected_status=(400,),
    )
    assert invalid.get("success") is False
    assert isinstance(invalid.get("errors"), list)
    print("AGRIVISION_PHASE1_API_SMOKE_OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000")
