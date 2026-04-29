# 实验七：胸部X光肺炎影像二分类
## 目录结构
hw07/
├── report.md # 实验报告
├── requirements.txt # 依赖
├── README.md # 本文档
└── figures/ # 图片文件夹
├── training_curves.png
└── confusion_matrix.png

## 运行方式

### 1. 安装依赖
```bash
pip install -r requirements.txt
2. 下载数据集
从 Kaggle 下载 Chest X-Ray Images (Pneumonia) 数据集
链接：https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
3. 运行训练
python train.py
数据集说明
训练集：5216张（80%训练 + 20%验证）
测试集：624张
类别：NORMAL（正常）、PNEUMONIA（肺炎）
实验结果           指标值
准确率 (Accuracy)	96.79%
精确率 (Precision)	97.94%
召回率 (Recall)	97.44%
F1分数	97.69%
环境要求
Python 3.8+
PyTorch 2.0+
CPU/GPU 均可

