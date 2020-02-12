# 动手学深度学习打卡笔记

by 干就完了-xihe

## Task 1：线性回归

### 1.线性回归的基本要素

​	模型

​	数据集

​	损失函数-MSE

​	优化函数-SG

### 2.向量计算

​	向量相加的方法：按元素逐一做标量加法 or 将两个向量直接矢量相加（运行更快）

### 3.线性回归的实现

​	步骤：

​	- 生成数据集

​	- 读取数据集

​	- 初始化模型参数

​	- 定义模型

​	- 定义损失函数

​	- 定义优化函数

​	- 模型训练

​	- 查验每个epoch的loss

------------------------------

## Task 2：softmax和分类模型

### 1.softmax的基本概念

 - 分类问题：将真实标签（eg.狗、猫、鸡等）对应为离散值如y1,y2,y3，进行模型的训练，最终根据新的特征来预测所属的类别
 - softmax回归同线性回归一样，也是一个单层神经网络，由于每个输出的计算都要依赖于所有的输入，他也是一个全连接层，不同的是，softmax的输出单元个数等于模型所要预测的类别个数，最终的输出为每个类别的置信度，其和为1

### 2.交叉熵损失函数

​	针对于分类任务，要想预测分类结果正确，并不需要预测概率完全等于标签概率，所以平方损失对于分类任务来说过于严格，改善此问题的一个方法就是使用更适合衡量两个概率分布差异的测量函数，其中交叉熵（cross entropy）是一个常用的测量方法。交叉熵只关心对正确类别的预测概率，也就是说只要其值足够大，就可以确保分类结果正确。



​	在训练好softmax回归模型后，给定任意样本特征，就可以预测每个输出类别的概率，通常我们把预测概率最大的类别作为输出类别，如果它与真是类别标签一致，说明这次预测是正确的。

### 3.Fashion-MNIST

1. torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
2. torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
3. torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
4. torchvision.utils: 其他的一些有用的方法。

### 4.模型的训练和预测

 - 获取训练集数据和测试集数据

 - 模型参数初始化

   ```
   batch_size = 256
   train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
   ```

 - 定义softmax操作

   ```
   def softmax(X):
       X_exp = X.exp()
       partition = X_exp.sum(dim=1, keepdim=True)
       return X_exp / partition
   ```

 - 定义softmax回归模型

   ```
   def net(X):
       return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
   ```

 - 定义损失函数-交叉熵

   ```
   def cross_entropy(y_hat, y):
       return - torch.log(y_hat.gather(1, y.view(-1, 1)))
   ```

 - 定义准确率

   ```
   def accuracy(y_hat, y):
       return (y_hat.argmax(dim=1) == y).float().mean().item()
   ```

 - 定义网络模型

   ```
   class LinearNet(nn.Module):
       def __init__(self, num_inputs, num_outputs):
           super(LinearNet, self).__init__()
           self.linear = nn.Linear(num_inputs, num_outputs)
       def forward(self, x): # x 的形状: (batch, 1, 28, 28)
           y = self.linear(x.view(x.shape[0], -1))
           return y
       
   # net = LinearNet(num_inputs, num_outputs)
   
   class FlattenLayer(nn.Module):
       def __init__(self):
           super(FlattenLayer, self).__init__()
       def forward(self, x): # x 的形状: (batch, *, *, ...)
           return x.view(x.shape[0], -1)
   
   from collections import OrderedDict
   net = nn.Sequential(
           # FlattenLayer(),
           # LinearNet(num_inputs, num_outputs) 
           OrderedDict([
              ('flatten', FlattenLayer()),
              ('linear', nn.Linear(num_inputs, num_outputs))])
   ```

 - 训练模型

 - 模型预测

​	