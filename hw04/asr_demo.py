#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音识别示例 - OpenAI Whisper
支持音频文件识别
"""

import whisper
import sys
import os

def transcribe_file(audio_path, model_size="base"):
    """
    识别音频文件
    """
    print("=" * 50)
    print("语音识别程序 - OpenAI Whisper")
    print("=" * 50)
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        return None
    
    print(f"📁 音频文件: {audio_path}")
    print(f"🔧 加载模型: {model_size}...")
    
    # 加载模型
    model = whisper.load_model(model_size)
    
    print(f"🎤 正在识别...")
    
    # 识别音频
    result = model.transcribe(audio_path, language="zh")
    
    print("\n" + "=" * 50)
    print("📝 识别结果:")
    print("=" * 50)
    print(result["text"])
    print("=" * 50)
    
    return result["text"]

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python asr_demo.py <音频文件路径>")
        print("示例: python asr_demo.py audio.mp3")
        return
    
    audio_file = sys.argv[1]
    transcribe_file(audio_file)

if __name__ == "__main__":
    main()
