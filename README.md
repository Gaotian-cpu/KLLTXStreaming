# LTX Streaming Interactive Engine

面向 **H100 / H200** 的 LTX-2.3 准实时流式视频生成服务。

输出格式为 **HLS (m3u8)**，支持边生成边播放。提供完整 HTTP 接口：提交任务、查询任务、获取播放流。

## 核心能力

- **HLS / m3u8 流式输出**：每个分片时长可配置，浏览器边下边播
- **任务接口**：提交 prompt + 图片 → 返回 task_id
- **查询接口**：轮询任务状态，就绪后返回 `stream_url`
- **播放流接口**：直接提供 m3u8 / ts，浏览器可播
- **动态提示词 / 动态 LoRA**（引擎层仍支持）
- **单卡 & 多卡**，优先蒸馏模型

## 目录结构

```
ltx_streaming_interactive/
├── api/server.py              # FastAPI 服务（任务 + HLS）
├── src/
│   ├── engine.py              # 生成引擎（含 run_task → HLS）
│   ├── task_manager.py        # 任务队列与状态
│   ├── hls/writer.py          # 帧 → .ts + m3u8
│   ├── prompt_manager.py
│   ├── lora_manager.py
│   └── ...
├── examples/
│   ├── interactive_cli.py     # 命令行交互
│   └── client_demo.py         # HTTP 客户端示例
├── configs/default.yaml
└── scripts/
    ├── run_api.sh             # 启动 API 服务
    ├── run_single.sh
    └── run_multi.sh
```

## 快速开始

```bash
# 1. 依赖
pip install -r requirements.txt
# 系统安装 ffmpeg（必须）
# Ubuntu: sudo apt install ffmpeg

# 2. 安装官方 LTX-2
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2 && pip install -e . && cd ..

# 3. 修改 configs/default.yaml 中的模型路径

# 4. 启动服务
bash scripts/run_api.sh
# 默认 http://0.0.0.0:8000
```

## HTTP 接口说明

### 1. 提交任务

```http
POST /api/v1/tasks
Content-Type: multipart/form-data

prompt: 一个女人正在唱歌
image: (可选文件)
segment_duration: 3.0      # 每个分片秒数，可配置
max_chunks: 20
width: 960                 # 可选
height: 544                # 可选
```

**响应示例：**

```json
{
  "task_id": "a1b2c3d4e5f67890",
  "status": "pending",
  "message": "task submitted",
  "query_url": "http://host:8000/api/v1/tasks/a1b2c3d4e5f67890"
}
```

### 2. 查询任务

```http
GET /api/v1/tasks/{task_id}
```

**响应示例（已有流可播）：**

```json
{
  "task_id": "a1b2c3d4e5f67890",
  "status": "streaming",
  "segment_count": 3,
  "progress": 0.15,
  "stream_url": "http://host:8000/streams/a1b2c3d4e5f67890/playlist.m3u8",
  "player_url": "http://host:8000/player/a1b2c3d4e5f67890"
}
```

状态含义：

| status     | 说明 |
|------------|------|
| pending    | 排队中 |
| running    | 正在生成第一段 |
| streaming  | 已有分片，可播放 |
| completed  | 全部生成完毕 |
| failed     | 失败（见 error 字段） |
| cancelled  | 已取消 |

### 3. 加载流（专用）

```http
GET /api/v1/tasks/{task_id}/stream
```

- 已就绪：返回 `stream_url`，`ready: true`
- 未就绪：HTTP 202，`ready: false`

### 4. 直接播放地址

```
GET /streams/{task_id}/playlist.m3u8
GET /streams/{task_id}/seg_00000.ts
...
GET /player/{task_id}          # 内置 hls.js 播放页
```

浏览器打开 `player_url` 即可边生成边看。

## 客户端示例

```bash
python examples/client_demo.py \
  --base http://127.0.0.1:8000 \
  --prompt "一个年轻女人深情唱歌，电影光效" \
  --image /path/to/face.png \
  --segment-duration 3 \
  --max-chunks 15
```

或用 curl：

```bash
# 提交
curl -X POST http://127.0.0.1:8000/api/v1/tasks \
  -F "prompt=一个女人在背唐诗" \
  -F "segment_duration=3" \
  -F "max_chunks=12" \
  -F "image=@face.png"

# 查询（把 task_id 换掉）
curl http://127.0.0.1:8000/api/v1/tasks/<task_id>
```

## 配置要点

`configs/default.yaml`：

```yaml
generation:
  fps: 24
  chunk_frames: 72          # 会被任务的 segment_duration 覆盖

server:
  host: "0.0.0.0"
  port: 8000
  public_base: ""           # 公网域名时填，例如 https://video.example.com
  max_workers: 1            # 并发任务数

hls:
  segment_duration: 3.0     # 默认分片时长（秒）
```

分片时长由提交任务时的 `segment_duration` 决定，引擎会按 `fps × duration` 计算每块帧数。

## 播放说明

- 桌面 Chrome / Edge / Firefox：用内置播放页（hls.js）
- Safari / iOS：原生支持 m3u8，可直接打开 `stream_url`
- VLC / ffplay：也可直接打开 m3u8 地址

生成第一个分片后状态变为 `streaming`，即可开始播放；后续分片会持续追加到同一 playlist。

## 注意

1. **必须安装 ffmpeg**（系统级），用于把帧编码成 `.ts`
2. 模型路径在 `configs/default.yaml` 的 `model.checkpoint`
3. 大模型场景下 `max_workers=1` 较稳妥；多卡可配合 `CUDA_VISIBLE_DEVICES` 起多个进程
4. `engine.py` 中 pipeline 加载逻辑需按你本地 LTX-2 / Diffusers 版本微调

## 旧版 CLI 交互

仍可使用：

```bash
bash scripts/run_single.sh
# 或
python examples/interactive_cli.py --config configs/default.yaml
```
