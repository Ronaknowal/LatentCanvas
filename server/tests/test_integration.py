# server/tests/test_integration.py
"""Integration smoke test — validates WebSocket flow with mocked pipeline."""
import io
import json
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image
from starlette.testclient import TestClient


def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def mock_pipeline():
    with patch("main.pipeline") as mock_pipe:
        mock_pipe.ready = True
        mock_pipe.model_config = MagicMock()
        mock_pipe.model_config.name = "test-model"
        mock_pipe.model_config.width = 512
        mock_pipe.model_config.height = 512
        mock_pipe.server_config = MagicMock()
        mock_pipe.server_config.default_prompt = "test"
        mock_pipe.server_config.default_strength = 0.5

        mock_pipe.generate_to_jpeg.return_value = _make_jpeg()
        yield mock_pipe


@pytest.fixture
def app():
    from main import app
    return app


def test_websocket_send_receive(app, mock_pipeline):
    """Send a sketch frame, receive a generated image back."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as ws:
            sketch = _make_jpeg()
            ws.send_bytes(sketch)

            # receive_bytes() blocks until the server sends a response
            response = None
            try:
                response = ws.receive_bytes()
            except Exception:
                pass

            assert response is not None
            assert len(response) > 0
            img = Image.open(io.BytesIO(response))
            assert img.format == "JPEG"


def test_websocket_config_update(app, mock_pipeline):
    """Send a JSON config update over WebSocket."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as ws:
            config = {"prompt": "anime style", "strength": 0.8}
            ws.send_text(json.dumps(config))

            ws.send_bytes(_make_jpeg())

            # receive_bytes() blocks until the server processes and responds
            try:
                ws.receive_bytes()
            except Exception:
                pass

            mock_pipeline.update_config.assert_called_with(
                prompt="anime style", strength=0.8
            )


def test_websocket_rejected_when_not_ready(app, mock_pipeline):
    """WebSocket should close if pipeline isn't ready."""
    mock_pipeline.ready = False

    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/generate") as ws:
                ws.receive_bytes()
