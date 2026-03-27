# 实验四：AI文稿生成 + 声音克隆 + 语音识别

## 目录结构
hw04/
├── text_gen.md # 任务一：AI生成文稿
├── jianying.md # 任务二：剪映声音克隆说明
├── asr_report.md # 任务三：ASR方案对比报告
├── experiment_log.md # 实验记录
├── asr_demo.py # 语音识别代码
├── requirements.txt # Python依赖
└── README.md # 本文档

## 运行说明

### 1. 安装依赖
```bash
pip install -r requirements.txt
python asr_demo.py C:\Users\吸氧羊\OneDrive\Desktop\voice_clone.mp3
各任务说明
任务一：AI生成文稿
使用 DeepSeek 生成文稿

详见 text_gen.md

任务二：剪映声音克隆
使用剪映内置音色生成配音

详见 jianying.md

任务三：语音识别调研与实现
对比了 Whisper、Vosk、FunASR 三种方案

选择 Whisper 本地实现

详见 asr_report.md 和 experiment_log.md
