"""Trusted protocol stub used only by ExternalCommandVerifier tests."""

import json
import sys


request = json.load(sys.stdin)
sample = request["sample"]
passed = sample["completion"] == sample["reference"]
json.dump(
    {
        "protocol": request["protocol"],
        "request_id": request["request_id"],
        "status": "pass" if passed else "wrong",
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "details": {"stub": True},
    },
    sys.stdout,
)
