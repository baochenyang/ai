# 实验三：人脸识别 Web 应用

## 项目简介
基于 `face_recognition` 和 `Streamlit` 构建的人脸检测与识别 Web 应用。用户上传图片后，系统自动检测人脸位置并可选识别身份。

## 功能说明
- **人脸检测**：检测图片中所有人脸，并用绿色框标出
- **人脸识别**：与已知人脸库比对，识别具体身份
- **Web界面**：Streamlit 提供的交互式界面，支持上传图片

## 运行环境
### 系统依赖（Windows）
安装 dlib 依赖（face_recognition 需要）：
```bash
pip install cmake
pip install dlib
