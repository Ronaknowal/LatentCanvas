import asyncio
import io
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import SD15_CONFIG, SERVER_CONFIG
from pipeline import RealtimeCanvasPipeline
from inference_queue import InferenceQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = RealtimeCanvasPipeline(
    model_config=SD15_CONFIG,
    server_config=SERVER_CONFIG,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize pipeline in a background thread
    logger.info("Starting pipeline initialization...")
    await asyncio.to_thread(pipeline.initialize)
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="LatentCanvas", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "ready" if pipeline.ready else "warming_up",
        "model": pipeline.model_config.name,
    }


@app.get("/api/config")
async def config():
    return {
        "model": pipeline.model_config.name,
        "width": pipeline.model_config.width,
        "height": pipeline.model_config.height,
        "default_prompt": pipeline.server_config.default_prompt,
        "default_strength": pipeline.server_config.default_strength,
    }


@app.websocket("/ws/generate")
async def generate_ws(websocket: WebSocket):
    await websocket.accept()

    if not pipeline.ready:
        await websocket.close(code=1013, reason="Server warming up")
        return

    queue = InferenceQueue()

    async def process_loop():
        while True:
            data = await queue.get_latest()
            if data is None:
                await asyncio.sleep(0.01)
                continue

            try:
                result = await asyncio.to_thread(pipeline.generate_to_jpeg, data)
                if result is not None:
                    await websocket.send_bytes(result)
            except Exception as e:
                logger.error(f"Generation error: {e}")

    process_task = asyncio.create_task(process_loop())

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                await queue.submit(message["bytes"])
            elif "text" in message and message["text"]:
                try:
                    config_data = json.loads(message["text"])
                    pipeline.update_config(
                        prompt=config_data.get("prompt"),
                        strength=config_data.get("strength"),
                    )
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON config")
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        process_task.cancel()
        try:
            await process_task
        except asyncio.CancelledError:
            pass
