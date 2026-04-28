#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
任务二：LeNet-5 手写数字识别
经典 LeNet-5 结构实现
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import time

print("TensorFlow 版本:", tf.__version__)
print("GPU 可用:", tf.config.list_physical_devices('GPU'))

# ==================== 1. 加载数据 ====================
print("\n" + "="*50)
print("1. 加载 MNIST 数据集")
print("="*50)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 归一化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# LeNet-5 原始输入是 32x32，需要将 28x28 缩放到 32x32
x_train = tf.image.resize(x_train[..., tf.newaxis], (32, 32)).numpy()
x_test = tf.image.resize(x_test[..., tf.newaxis], (32, 32)).numpy()

print(f"训练集大小: {x_train.shape}")
print(f"测试集大小: {x_test.shape}")

# ==================== 2. 构建 LeNet-5 模型 ====================
print("\n" + "="*50)
print("2. 构建 LeNet-5 模型")
print("="*50)

def create_lenet5():
    """创建 LeNet-5 模型"""
    model = models.Sequential([
        # C1: 卷积层, 6个 5x5 卷积核
        layers.Conv2D(6, (5, 5), activation='tanh', padding='valid', input_shape=(32, 32, 1)),
        # S2: 平均池化层
        layers.AveragePooling2D((2, 2), strides=2),
        
        # C3: 卷积层, 16个 5x5 卷积核
        layers.Conv2D(16, (5, 5), activation='tanh'),
        # S4: 平均池化层
        layers.AveragePooling2D((2, 2), strides=2),
        
        # C5: 卷积层, 120个 5x5 卷积核
        layers.Conv2D(120, (5, 5), activation='tanh'),
        
        # 展平
        layers.Flatten(),
        
        # F6: 全连接层, 84个神经元
        layers.Dense(84, activation='tanh'),
        
        # 输出层: 10个类别
        layers.Dense(10, activation='softmax')
    ])
    return model

model = create_lenet5()
model.summary()

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
epochs = 10

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
model.save('lenet5_mnist.h5')
print("\n模型已保存为: lenet5_mnist.h5")

# ==================== 7. 输出结果汇总 ====================
print("\n" + "="*50)
print("6. 结果汇总")
print("="*50)
print(f"训练轮数: {epochs}")
print(f"批量大小: {batch_size}")
print(f"优化器: Adam")
print(f"激活函数: tanh (原始 LeNet-5 风格)")
print(f"训练耗时: {train_time:.2f} 秒")
print(f"模型参数量: {total_params:,}")
print(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
