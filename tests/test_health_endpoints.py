import pytest


@pytest.mark.asyncio
async def test_health_endpoint_works(client):
    resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
