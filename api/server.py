#!/usr/bin/env python3
"""
LTX Streaming Service — HTTP API

接口：
  POST   /api/v1/tasks              提交生成任务（prompt + 可选图片）
  GET    /api/v1/tasks/{task_id}    查询任务状态；有流则返回 stream_url
  GET    /api/v1/tasks              列出最近任务
  POST   /api/v1/tasks/{task_id}/cancel  取消任务
  GET    /streams/{task_id}/playlist.m3u8   HLS 播放列表
  GET    /streams/{task_id}/seg_XXXXX.ts    HLS 分片
  GET    /player/{task_id}          简易浏览器播放页
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from PIL import Image
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_model_paths
from src.engine import StreamingEngine
from src.task_manager import TaskManager, TaskStatus
from src.utils import setup_logger

logger = setup_logger("api")

# ---------------------------------------------------------------------------
# 全局
# ---------------------------------------------------------------------------
CFG_PATH = ROOT / "configs" / "default.yaml"
def _parse_startup_args():
    """uvicorn 直接加载模块时也能从环境变量读；python -m api.server 时用 CLI。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-root",
        type=str,
        default=os.environ.get("LTX_MODEL_ROOT", ""),
        help="模型文件根目录，与 yaml 中相对路径拼接",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CFG_PATH),
        help="配置文件路径",
    )
    args, _ = parser.parse_known_args()
    return args


_startup = _parse_startup_args()
cfg = load_config(_startup.config)
cfg = resolve_model_paths(cfg, model_root=_startup.model_root or None)

# 可选：打印确认
import logging
logging.getLogger("api").info(
    "model.checkpoint=%s text_encoder=%s spatial_upsampler=%s",
    cfg.model.get("checkpoint"),
    cfg.model.get("text_encoder"),
    cfg.model.get("spatial_upsampler"),
)

STREAMS_ROOT = Path(cfg.streaming.get("output_dir", "outputs")) / "streams"
STREAMS_ROOT.mkdir(parents=True, exist_ok=True)

# 对外访问的 base（可用环境变量覆盖）
PUBLIC_BASE = cfg.get("server", {}).get("public_base", "")

task_manager = TaskManager(
    streams_root=STREAMS_ROOT,
    max_workers=int(cfg.get("server", {}).get("max_workers", 1)),
)


def _engine_factory() -> StreamingEngine:
    # 每个任务可以新建引擎；大模型场景下也可改成单例 + 锁
    return StreamingEngine(cfg)


task_manager.set_engine_factory(_engine_factory)

app = FastAPI(
    title="LTX Streaming Interactive API",
    version="1.0.0",
    description="提交提示词/图片 → 流式生成 m3u8 → 浏览器直接播放",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态挂载 HLS 目录（playlist + ts）
app.mount("/streams", StaticFiles(directory=str(STREAMS_ROOT)), name="streams")


def _base_url(request: Request) -> str:
    if PUBLIC_BASE:
        return PUBLIC_BASE.rstrip("/")
    # 自动从请求推断
    return str(request.base_url).rstrip("/")


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    task_manager.start_workers()
    logger.info("API server started, workers running.")


@app.on_event("shutdown")
def on_shutdown():
    task_manager.stop()


# ---------------------------------------------------------------------------
# 任务接口
# ---------------------------------------------------------------------------
@app.post("/api/v1/tasks")
async def submit_task(
    request: Request,
    prompt: str = Form(..., description="生成提示词"),
    image: Optional[UploadFile] = File(None, description="可选初始图片（I2V）"),
    segment_duration: float = Form(3.0, description="每个 HLS 分片时长（秒）"),
    max_chunks: int = Form(20, description="最多生成多少个分片"),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
):
    """
    提交生成任务。
    返回 task_id，随后用查询接口轮询；出现 stream_url 后即可播放。
    """
    if not prompt or not prompt.strip():
        raise HTTPException(400, "prompt is required")

    pil_image = None
    if image is not None and image.filename:
        data = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"invalid image: {e}")

    # 限制范围
    segment_duration = max(1.0, min(segment_duration, 10.0))
    max_chunks = max(1, min(max_chunks, 120))

    info = task_manager.submit(
        prompt=prompt.strip(),
        image=pil_image,
        segment_duration=segment_duration,
        max_chunks=max_chunks,
        width=width,
        height=height,
    )
    base = _base_url(request)
    return JSONResponse(
        status_code=201,
        content={
            "task_id": info.task_id,
            "status": info.status.value,
            "message": "task submitted",
            "query_url": f"{base}/api/v1/tasks/{info.task_id}",
        },
    )


@app.get("/api/v1/tasks/{task_id}")
def query_task(task_id: str, request: Request):
    """
    查询任务状态。
    当 status 为 streaming / completed 时，stream_url 可用，浏览器直接播。
    """
    info = task_manager.get(task_id)
    if not info:
        raise HTTPException(404, "task not found")
    base = _base_url(request)
    data = info.to_dict(base_url=base)
    # 额外给一个播放页
    if info.stream_url:
        data["player_url"] = f"{base}/player/{task_id}"
    return data


@app.get("/api/v1/tasks")
def list_tasks(request: Request, limit: int = 50):
    base = _base_url(request)
    items = [t.to_dict(base_url=base) for t in task_manager.list_tasks(limit=limit)]
    return {"tasks": items}


@app.post("/api/v1/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    ok = task_manager.cancel(task_id)
    if not ok:
        raise HTTPException(400, "cannot cancel (not found or already finished)")
    return {"task_id": task_id, "status": "cancelled"}


@app.get("/api/v1/tasks/{task_id}/stream")
def get_stream_info(task_id: str, request: Request):
    """
    专门的「加载流」接口：只返回播放相关信息。
    若尚未有分片，返回 202。
    """
    info = task_manager.get(task_id)
    if not info:
        raise HTTPException(404, "task not found")
    base = _base_url(request)
    if info.status in (TaskStatus.STREAMING, TaskStatus.COMPLETED) and info.stream_url:
        url = info.stream_url
        if not url.startswith("http"):
            url = f"{base}/{url.lstrip('/')}"
        return {
            "task_id": task_id,
            "status": info.status.value,
            "stream_url": url,
            "player_url": f"{base}/player/{task_id}",
            "segment_count": info.segment_count,
            "ready": True,
        }
    return JSONResponse(
        status_code=202,
        content={
            "task_id": task_id,
            "status": info.status.value,
            "stream_url": None,
            "ready": False,
            "message": "stream not ready yet",
        },
    )


# ---------------------------------------------------------------------------
# 简易播放页（浏览器打开即可播 m3u8）
# ---------------------------------------------------------------------------
@app.get("/player/{task_id}", response_class=HTMLResponse)
def player_page(task_id: str, request: Request):
    info = task_manager.get(task_id)
    if not info:
        raise HTTPException(404, "task not found")
    base = _base_url(request)
    stream = f"{base}/streams/{task_id}/playlist.m3u8"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>LTX Stream — {task_id}</title>
  <script src="https://cdn.jsdelivr.net/npm/hls.js@1.5.7/dist/hls.min.js"></script>
  <style>
    body {{ margin:0; background:#111; color:#eee; font-family:system-ui,sans-serif; }}
    .wrap {{ max-width:960px; margin:24px auto; padding:0 12px; }}
    video {{ width:100%; background:#000; border-radius:8px; }}
    .meta {{ margin-top:12px; font-size:14px; opacity:0.85; }}
    code {{ background:#222; padding:2px 6px; border-radius:4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>LTX Streaming Player</h2>
    <video id="v" controls autoplay muted playsinline></video>
    <div class="meta">
      <div>Task: <code>{task_id}</code></div>
      <div>Status: <code id="st">{info.status.value}</code></div>
      <div>Stream: <code>{stream}</code></div>
    </div>
  </div>
  <script>
    const src = "{stream}";
    const video = document.getElementById("v");
    function start() {{
      if (Hls.isSupported()) {{
        const hls = new Hls({{ enableWorker: true, lowLatencyMode: true }});
        hls.loadSource(src);
        hls.attachMedia(video);
        hls.on(Hls.Events.MANIFEST_PARSED, () => video.play().catch(()=>{{}}));
        hls.on(Hls.Events.ERROR, (_, data) => {{
          if (data.fatal) setTimeout(() => {{ hls.loadSource(src); }}, 2000);
        }});
      }} else if (video.canPlayType("application/vnd.apple.mpegurl")) {{
        video.src = src;
        video.addEventListener("loadedmetadata", () => video.play().catch(()=>{{}}));
      }}
    }}
    start();
    // 简单轮询状态
    setInterval(async () => {{
      try {{
        const r = await fetch("{base}/api/v1/tasks/{task_id}");
        const j = await r.json();
        document.getElementById("st").textContent = j.status;
      }} catch(e) {{}}
    }}, 3000);
  </script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------------------------
# 本地启动
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # 再解析一次，保证 --model-root 生效（模块 import 时已处理过一遍也可）
    startup = _parse_startup_args()
    # cfg 已在模块级 resolve；若希望严格以 CLI 为准，可再 resolve 一次：
    # global cfg
    # cfg = resolve_model_paths(load_config(startup.config), startup.model_root or None)

    host = cfg.get("server", {}).get("host", "0.0.0.0")
    port = int(cfg.get("server", {}).get("port", 8000))
    uvicorn.run("api.server:app", host=host, port=port, reload=False)
