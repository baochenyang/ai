#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
任务一：极简 CNN 手写数字识别
参考：公众号文章《计算机视觉》第10篇
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import time

# 打印版本信息
print("TensorFlow 版本:", tf.__version__)
print("GPU 可用:", tf.config.list_physical_devices('GPU'))

# ==================== 1. 加载数据 ====================
print("\n" + "="*50)
print("1. 加载 MNIST 数据集")
print("="*50)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 归一化到 [0,1] 范围
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 重塑为 (28, 28, 1) 格式（CNN 需要通道维度）
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"训练集大小: {x_train.shape}")
print(f"测试集大小: {x_test.shape}")

# ==================== 2. 构建极简 CNN 模型 ====================
print("\n" + "="*50)
print("2. 构建极简 CNN 模型")
print("="*50)

model = models.Sequential([
    # 第一层卷积 + 池化
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # 第二层卷积 + 池化
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 第三层卷积
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 展平 + 全连接层
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 显示模型结构
model.summary()

# 计算参数量
total_params = model.count_params()
print(f"\n模型参数量: {total_params:,}")

# ==================== 3. 编译模型 ====================
print("\n" + "="*50)
print("3. 编译模型")
print("="*50)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==================== 4. 训练模型 ====================
print("\n" + "="*50)
print("4. 开始训练")
print("="*50)

batch_size = 64
epochs = 5

start_time = time.time()

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=1
)

train_time = time.time() - start_time
print(f"\n训练耗时: {train_time:.2f} 秒")

# ==================== 5. 评估模型 ====================
print("\n" + "="*50)
print("5. 模型评估")
print("="*50)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"测试集损失: {test_loss:.4f}")

# ==================== 6. 保存模型 ====================
model.save('simple_cnn_mnist.h5')
print("\n模型已保存为: simple_cnn_mnist.h5")

# ==================== 7. 输出结果汇总 ====================
print("\n" + "="*50)
print("6. 结果汇总")
print("="*50)
print(f"训练轮数: {epochs}")
print(f"批量大小: {batch_size}")
print(f"优化器: Adam")
print(f"损失函数: sparse_categorical_crossentropy")
print(f"训练耗时: {train_time:.2f} 秒")
print(f"模型参数量: {total_params:,}")
print(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
