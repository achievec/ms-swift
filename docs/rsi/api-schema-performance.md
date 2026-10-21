# RSI `/api/schema` 首屏慢的定位与优化建议

现象：DevTools 里 `GET /api/schema` 出现 **1.7s** 量级的水瀑布，且 RSI 前端 `boot()` 是先把 schema 拿到再渲染，所以这段时间是白屏。

本文先做定位 + 优化思路；**P0 已于 2026-10-21 落地**（GPU TTL/stale-on-error、`/api/health` 解耦、预序列化/预压缩/`ETag`/`304`、构建单飞）。P1/P2 仍为建议。实测基线对象：`http://10.164.162.4:8418`（含 `9077549f` 的 dict 缓存 + `GZipMiddleware` + 布局去重）。

---

## 1. 实测数据

### 1.1 线上各接口耗时（keep-alive 连接，curl/http.client）

| 请求 | 响应体 | TTFB | 备注 |
| --- | --- | --- | --- |
| `GET /nope`（404） | 22 B | 73–166 ms | **基线**：几乎空响应也要 70ms+ |
| `GET /static/index.html` | 2.2 KB | 75–154 ms | 同上 |
| `GET /api/runs` | 269 B | 74–330 ms | 抖动很大 |
| `GET /api/health` | 335 B | 130–224 ms | 基线 + 多出来的 `nvidia-smi` |
| `GET /api/schema`（gzip） | 34 KB | 230–386 ms | |
| `GET /api/schema`（不压缩） | 311 KB | TTFB 216–332 ms，总 590–690 ms | |

关键对比：**335 字节的 `/api/health` 比 22 字节的 404 还慢 60–90ms**，说明慢的不是"查数据"，而是每个请求固定附带的一笔开销；而 311KB 的 schema 在缓存命中的情况下，服务端处理只需要几毫秒（见下表）。

### 1.2 本地同一份代码的分段耗时（`rsi/schemas.py`）

| 环节 | 耗时 |
| --- | --- |
| `detect_gpu()`（Mac 无 `nvidia-smi`，直接返回） | 0.4 ms |
| `model_source()`（每次读 `configs/fetch.yaml`） | 1.2 ms |
| `public_schema()` **冷构建** | **3577 ms** |
| `public_schema()` 命中缓存 | 1.0 ms |
| `jsonable_encoder`（335KB dict） | 7.8 ms |
| `json.dumps`（335KB） | 1.3 ms |
| `gzip.compress`（335KB → 36KB） | 2.4 ms |
| `export_layout()` 首次调用（`llm_export` 布局） | 705 ms |

结论：**命中缓存后服务端只花几毫秒**；慢的是「冷构建」和「每个请求的固定旁路开销」，再加上这台机器本身的事件循环基线。

---

## 2. 慢在哪（按影响排序）

### P0-1 每个请求都 fork 一次 `nvidia-smi`，还拿它当缓存 key

```python
def public_schema() -> dict:
    device_count = len((detect_gpu().get('devices') or []))   # 每次请求都 subprocess.run(['nvidia-smi', '-L'])
    key = (device_count, model_source())
    ...
```

- `detect_gpu()` 每次都起子进程，空闲卡几十 ms，**训练把卡占满时可能几百 ms 到数秒**（`timeout=5`，超时就直接返回空列表）。
- `/api/health` 里还会再来一次，等于一次健康检查探测两遍。
- 更严重的是**抖动 → 缓存击穿**：只要某次探测返回空（超时/驱动抖动/`nvidia-smi` 被卡住），key 就从 `(8, 'modelscope')` 变成 `(0, 'modelscope')`，缓存未命中 → **整份 3.5s 的表单树重建**。这就是 DevTools 里"平时 300ms、偶尔 1.7s"的来源，而不是接口逻辑本身慢。

### P0-2 `/api/health` 为了一个 `hardware` 字段构建整份 schema

```python
def health():
    schema = public_schema()          # 冷进程上 3.5s
    return {..., 'hardware': schema.get('hardware') or {}}
```

`hardware` 就在 `configs/a100_80g.yaml` 里（`presets()['hardware']`），根本不需要表单树。冷启动后任何一次 health 轮询都会把 3.5s 账单先记在这条 335 字节的请求上。

### P1-3 每个请求重新序列化 + 重新压缩 335KB

`return public_schema()`（dict）会走 FastAPI 的 `jsonable_encoder` → `json.dumps`，然后 `GZipMiddleware` 在**事件循环里**再压一遍。单次约 10ms CPU，看起来不多，但它压在一次"大响应"的关键路径上，和所有其他请求抢同一个事件循环——这也是旁边 `/api/runs` 会突然 330ms 的原因之一。

### P1-4 没有 `ETag` / `304`

每次刷新页面都重新下载 335KB（gzip 36KB）。RSI 是"反复改参数、反复刷新"的用法，这一份流量本该在第二次之后变成 0 字节。

### P2-5 冷构建落在"第一个用户请求"上

`public_schema()` 的 3.5s 只在进程启动后第一次被调用时发生，而它恰好是首屏那条请求。前端 `boot()`：

```js
async function boot() {
  state.schema = await api('/api/schema');   // 3.5s 冷构建时，这里一直等
  fillCreateModel();
  await refreshRuns();
  render();
}
```

白屏时间 ≈ 冷构建时间。

### P2-6 首屏串行

`schema` 和 `/api/runs` 互不依赖，却串行发了两次往返。

### P2-7 运行环境基线：这台机器/进程本身已经被拖慢

`404` 和静态文件也要 73–155ms（正常同机房应该是 1–3ms），说明进程里/机器上一直有别的东西在抢 CPU 和事件循环：

- 训练任务本身（`swift` 子进程、数据加载）在同一台机器上跑；
- 日志 tail、hope 拉取、serve 探活等后台线程；
- 前端每 2s 一轮的 live 轮询（`/api/runs/{id}`、`log`、`metrics`）。

另外，`rsi/app.py` 里有若干 `async def` 端点内部直接调用同步的重活（例如非流式 `/chat`、`preview`、`export`），**会把事件循环阻塞住**，一个慢请求连累所有请求。这块不属于 `/api/schema` 自身，但会让"整体都慢半拍"，建议一起处理。

---

## 3. 优化思路（按性价比）

### P0：先把"命中路径"压到毫秒级

**① GPU 探测加 TTL 缓存 + stale-on-error** — ✅ 已完成（2026-10-21）

实现：`rsi.backends.cached_gpu`（TTL 30s + 探测为空时保留上次成功结果）。`schema` 与 `health` 都走它。

```python
_GPU_TTL = 30.0
_gpu_cache = {'at': float('-inf'), 'value': None}

def cached_gpu(ttl: float = _GPU_TTL) -> dict:
    now = time.monotonic()
    last = _gpu_cache['value']
    if last is not None and now - _gpu_cache['at'] < ttl:
        return last
    fresh = detect_gpu()
    if not fresh.get('devices') and last and last.get('devices'):
        fresh = last          # 探测抖动时保留上一次成功结果，别让 key 翻转
    _gpu_cache['value'], _gpu_cache['at'] = fresh, now
    return fresh
```

收益：去掉每请求 60–200ms，并彻底消除"探测抖动 → 3.5s 重建"。

**② `/api/health` 只读预设** — ✅ 已完成（2026-10-21）

```python
'hardware': presets().get('hardware') or {},
```

收益：health 从"可能 3.5s"变成 <1ms，冷启动时的首屏不再被它抢跑。

**③ schema 预序列化 + 预压缩 + `ETag`** — ✅ 已完成（2026-10-21）

实现：`rsi.schemas.public_schema_wire()` + `/api/schema` 返回 `Response(bytes)`。

```python
def public_schema_wire() -> tuple:          # (raw, gzipped, etag)，每次构建只算一次
    raw = json.dumps(public_schema(), ensure_ascii=False, separators=(',', ':')).encode()
    return raw, gzip.compress(raw, 6), '"' + hashlib.blake2b(raw, digest_size=16).hexdigest() + '"'
```

端点侧：

- `If-None-Match` 命中 → `304`（约 1ms、0 字节）；
- `Accept-Encoding: gzip` → 直接回预压缩字节并带 `Content-Encoding: gzip`（`GZipMiddleware` 见到 `Content-Encoding` 会原样透传，不会二次压缩；`Vary` 需要自己补）。

注意：返回 `Response(bytes)` 而不是 dict，可以顺带跳过 `jsonable_encoder`。

**④ 构建单飞（single-flight）** — ✅ 已完成（2026-10-21）

加一把 `threading.Lock`，双重检查缓存：并发首屏只构建一次，其余请求等这一次结果，而不是各自构建 3.5s。

### P1：消掉首屏的 3.5s 与一次往返

**⑤ 启动预热**：在 `rsi/__main__.main()` 里起一个 daemon 线程调 `public_schema()`。注意别放进 `lifespan`，否则每个用 `TestClient` 的测试都会多付一次 3.5s；也不要放在 `mount_swift_ui()` 之前，避免和 Gradio 挂载抢 CPU。

**⑥ 前端并行**：

```js
const [schema] = await Promise.all([api('/api/schema'), refreshRuns()]);
```

**⑦ schema 瘦身**（进一步降体积）：当前 335KB 的大头是 `catalog.models`、`catalog.by_model` 和各 stage 的完整布局。可选做法：

- 首屏只下发当前 stage 的 layout，其余 stage 在切 tab 时按需取 `/api/schema?stage=rl`；
- `by_model` 从 `{id: {model_type, template}}` 压成 `{id: "qwen2/qwen2_5"}`（体积约减半）；
- 目标是 raw < 100KB / gzip < 15KB。

### P2：环境层面（决定"整体快不快"）

**⑧** 把 `async def` 端点里的同步重活挪进线程池（`await run_in_threadpool(pipeline.xxx, ...)`），至少覆盖 `/chat`（非流式）、`preview`、`export`、`fetch`；SQLite 写入、扫盘、`subprocess` 都不要在事件循环里做。
**⑨** 前端 live 轮询降频/合并：现在 2s 一轮拉 view + log + metrics，训练中三条都很大，可以合并成一个 `/api/runs/{id}/live?since=...` 增量接口。
**⑩** 让 `detect_gpu()` 的后台刷新独立于请求路径（例如启动时探测一次 + 30s 定时刷新），请求路径永远读缓存值。

---

## 4. 验证方式

```bash
# 命中缓存的 TTFB / 总耗时
curl -s -o /dev/null -H 'Accept-Encoding: gzip' \
  -w 'ttfb=%{time_starttransfer}s total=%{time_total}s size=%{size_download}\n' \
  http://10.164.162.4:8418/api/schema

# 带 ETag 重放，期望 304 且 size=0
curl -s -D- -o /dev/null -H 'Accept-Encoding: gzip' \
  -H 'If-None-Match: "<上一步返回的 etag>"' \
  http://10.164.162.4:8418/api/schema
```

期望结果：

| 场景 | 现状 | 优化后 |
| --- | --- | --- |
| 冷启动后首个请求 | ~1.7–3.5s | 预热完成则 ~基线，否则等预热剩余时间 |
| 缓存命中 | 230–390ms TTFB（含 `nvidia-smi` + 序列化 + 压缩） | 基线 + ~2ms |
| 刷新页面（ETag 命中） | 重新下 36KB | `304`，0 字节 |
| 训练中偶发尖峰 | 1.7s（key 翻转重建） | 无（探测抖动不再触发重建） |

---

## 5. 一句话结论

`/api/schema` 慢**不是**因为查库慢，而是：

1. 每个请求都 fork `nvidia-smi`，并且拿它的结果当缓存 key——探测一抖动就触发 **3.5s 全量重建**；
2. `/api/health` 为了 `hardware` 一个字段也去构建整份 schema；
3. 335KB 大 JSON 每请求重新序列化/压缩，且没有 `ETag`，刷新永远重下 36KB。

先做 P0 三条（GPU 缓存 + health 解耦 + 预压缩/304），把命中路径压到毫秒级；再用启动预热和前端并行把首屏那 3.5s 抹掉；最后处理 async 端点里的同步阻塞，把整台机器的基线从 70ms+ 拉回正常水平。
