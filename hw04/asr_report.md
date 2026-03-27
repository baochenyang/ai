# 任务三：开源语音识别（ASR）方案调研

## 一、方案对比

| 对比维度 | OpenAI Whisper | Vosk | FunASR |
|----------|----------------|------|--------|
| **开发机构** | OpenAI | Alpha Cephei | 阿里达摩院 |
| **许可协议** | MIT | Apache 2.0 | MIT |
| **语言支持** | 99+ 语言，中文支持好 | 20+ 语言，中文支持 | 专注中文 |
| **模型体量** | tiny(39M) ~ large(1.5G) | 40M ~ 1.2G | 220M ~ 1.2G |
| **推理速度** | 较慢（GPU加速） | 较快（CPU即可） | 中等 |
| **实时/流式** | 不支持 | 支持 | 支持 |
| **部署难度** | 简单（pip安装） | 简单 | 中等 |
| **准确率** | 高 | 中等 | 高（中文） |

## 二、选型理由

我选择 **OpenAI Whisper**，理由如下：
1. **准确率高**：Whisper 在多种语言上表现优异
2. **安装简单**：`pip install openai-whisper` 即可
3. **支持中文**：对中文语音识别效果不错
4. **社区活跃**：文档完善，问题容易解决

## 三、本地实现

### 环境配置
- **OS**: Windows 11
- **Python**: 3.10
- **GPU**: 无（CPU运行）

### 安装依赖
```bash
pip install openai-whisper
pip install torch  # 如果安装失败，可安装CPU版

### 运行代码
```python
import whisper

# 加载模型（tiny/small/base/large）
model = whisper.load_model("base")

# 识别音频文件
result = model.transcribe("audio.mp3", language="zh")
print(result["text"])

### 运行结果示例


### 遇到的问题与解决
- **问题**：torch 安装失败
- **解决**：使用 `pip install torch --index-url https://download.pytorch.org/whl/cpu` 安装 CPU 版本
