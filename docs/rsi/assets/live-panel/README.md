# live-panel 示意图源文件

用 [live-panel](https://github.com/ythx-101/live-panel-skill) 从 JSON 渲染出可循环 HTML / mp4 / 预览图，嵌入用户手册与架构文档。

| 目录 | 对应文档中的图 | 说明 |
|------|----------------|------|
| `flow-map/` | [2026-10-23-user-guide.md](../../2026-10-23-user-guide.md) 功能地图 | 浅色 scaffold：主链路 + 旁路能力 rail |
| `architecture/` | [2026-10-23-architecture-ops.md](../../2026-10-23-architecture-ops.md) 运行时分层 | 终端深色 scaffold：浏览器→…→磁盘 |

重新生成（本机需 Chrome + ffmpeg；skill 已装在 `~/.agents/skills/live-panel`）：

```bash
SKILL="$HOME/.agents/skills/live-panel"
ROOT="docs/rsi/assets/live-panel"

python3 "$SKILL/scripts/scaffold.py" --spec "$ROOT/flow-map/spec.json" --out "$ROOT/flow-map/config.json"
python3 "$SKILL/scripts/check_frames.py" --config "$ROOT/flow-map/config.json" --out-dir /tmp/lp-flow --repeat
python3 "$SKILL/scripts/render.py" --config "$ROOT/flow-map/config.json" \
  --out "$ROOT/flow-map/flow-map.mp4" --html-out "$ROOT/flow-map/flow-map.html"

python3 "$SKILL/scripts/scaffold.py" --spec "$ROOT/architecture/spec.json" --out "$ROOT/architecture/config.json"
python3 "$SKILL/scripts/check_frames.py" --config "$ROOT/architecture/config.json" --out-dir /tmp/lp-arch --repeat
python3 "$SKILL/scripts/render.py" --config "$ROOT/architecture/config.json" \
  --out "$ROOT/architecture/architecture.mp4" --html-out "$ROOT/architecture/architecture.html"
```

动图数字为示意，非真实实验遥测。文档里仍保留 mermaid，便于纯文本 diff / 无资源预览。
