import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from PIL import Image
import io


# Mock the pipeline before importing main
@pytest.fixture(autouse=True)
def mock_pipeline():
    """Mock the GPU pipeline for all tests."""
    with patch("main.pipeline") as mock_pipe:
        mock_pipe.ready = True
        mock_pipe.model_config = MagicMock()
        mock_pipe.model_config.name = "test-model"
        mock_pipe.server_config = MagicMock()
        mock_pipe.server_config.default_prompt = "test prompt"
        mock_pipe.server_config.default_strength = 0.5

        # generate_to_jpeg returns a small JPEG
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        mock_pipe.generate_to_jpeg.return_value = buf.getvalue()

        yield mock_pipe


@pytest.fixture
def app():
    from main import app
    return app


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_ready(self, app, mock_pipeline):
        mock_pipeline.ready = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    async def test_health_warming_up(self, app, mock_pipeline):
        mock_pipeline.ready = False
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "warming_up"


@pytest.mark.asyncio
class TestConfigEndpoint:
    async def test_config_returns_model_info(self, app, mock_pipeline):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "default_prompt" in data
        assert "default_strength" in data
