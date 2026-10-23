# RSI：压缩包数据集上传

上传来源除 `.jsonl` / `.json` / `.txt` 外，还支持 `.zip`、`.tar`、`.tar.gz`、`.tgz`。服务端会安全解压，并把相对媒体路径改成解压目录下的绝对路径，再交给现有的 `--dataset` 文件路径流程。上传体积不设上限。

## 打包方式

推荐目录结构：

```text
my_dataset.zip
├── data.jsonl          # 或 train.jsonl；多个 jsonl 时优先这两个名字，否则取最大的
└── images/
    └── a.jpg
```

`data.jsonl` 里用相对路径指向媒体，例如：

```json
{"messages":[{"role":"user","content":"<image>描述图片"},{"role":"assistant","content":"..."}],"images":["images/a.jpg"]}
```

也支持字段别名 `image` / `video` / `audio`，以及 `images` / `videos` / `audios`；值为字符串、字符串列表，或嵌套列表（如视频帧列表）。消息里的 `<image>` 等占位符保持不动。

以下路径**不会**改写：

- 已是绝对路径
- `http://` / `https://` / `data:` URI

若改写后的本地文件不存在，校验结果里会带警告，但不会仅因此失败。

## 安全

解压会拒绝路径穿越（`../`）、绝对路径成员，以及指向解压目录外的符号链接。产物落在 `$RSI_HOME/uploads/<uuid>/`，训练用的是其中的 `resolved.jsonl`（若相对路径已全部是绝对路径且源文件已是 jsonl，则直接指向该源文件）。
