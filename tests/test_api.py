import os

os.environ["LLM_PROVIDER"] = "rules"
os.environ["RATE_LIMIT_PER_HOUR"] = "1000"

from fastapi.testclient import TestClient  # noqa: E402

from health_agent.server import app  # noqa: E402


def test_chat_flow_health_and_stats():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "healthy"

        r = client.post("/chat", json={"message": "I'm 45, sugar 160", "visitor_id": "browser-a"}).json()
        sid = r["session_id"]
        assert r["reply"]["status"] == "need_info" and r["reply"]["missing_fields"] == ["bmi"]

        r = client.post("/chat", json={"session_id": sid, "message": "my BMI is 29", "visitor_id": "browser-a"}).json()
        assert r["reply"]["status"] == "result"
        assert r["reply"]["tier"] and r["reply"]["model_version"]
        assert r["reply"]["disclaimer"].startswith("Educational demo")

        stats = client.get("/stats").json()
        assert stats["messages"] >= 2 and stats["sessions"] >= 1 and stats["tool_calls"] >= 1
        assert stats["unique_visitors"] >= 1 and stats["visitors_with_a_result"] >= 1
        assert "browser-a" not in str(stats)  # only counts are exposed


def test_validation_errors():
    with TestClient(app) as client:
        assert client.post("/chat", json={"message": ""}).status_code == 422
        assert client.post("/chat", json={"message": "x" * 1001}).status_code == 422


def test_rate_limit():
    with TestClient(app) as client:
        app.state.limiter.per_hour = 2
        try:
            codes = [client.post("/chat", json={"message": "hi"}).status_code for _ in range(3)]
        finally:
            app.state.limiter.per_hour = 1000
        assert codes == [200, 200, 429]
