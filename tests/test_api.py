from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
