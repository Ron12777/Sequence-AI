import pytest
from web import app as web


@pytest.fixture
def client():
    web.mcts = None
    return web.app.test_client()


def test_validation(client):
    assert client.post("/api/new_game", json=[]).status_code == 400
    assert client.post("/api/new_game", json={"ai_player": 3}).status_code == 400
    assert client.post("/api/new_game", json={}).status_code == 200
    assert client.post("/api/ai_move", json={"simulations": "bad"}).status_code == 400


def test_greedy_fallback_completes_and_clears_status(client):
    client.post("/api/new_game", json={"ai_player": 1})
    response = client.post("/api/ai_move", json={})
    assert response.status_code == 200
    assert response.json["ai_move"] is not None
    assert web.ai_status["thinking"] is False
