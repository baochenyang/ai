# 调试记录

## 问题1：数据形状不匹配

**现象**：ValueError: Input 0 of layer conv2d is incompatible with the layer: expected ndim=4, found ndim=3.

**原因分析**：
MNIST 原始数据形状是 (28, 28)，没有通道维度。CNN 需要 (height, width, channels) 格式。

**解决方案**：
```python
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)问题2：LeNet-5 输入尺寸不匹配
现象：
LeNet-5 期望输入 32×32，但 MNIST 是 28×28。

原因分析：
经典 LeNet-5 设计用于 32×32 输入。

解决方案：
x_train = tf.image.resize(x_train[..., tf.newaxis], (32, 32)).numpy()
x_test = tf.image.resize(x_test[..., tf.newaxis], (32, 32)).numpy()
