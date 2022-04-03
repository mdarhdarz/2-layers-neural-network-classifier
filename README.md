## MNIST数据集，两层全连接神经网络分类器识别手写数字

#### 所使用python库

numpy 1.20.3

scikit-learn 1.0.1

matplotlib 3.4.3

#### 训练及测试步骤

运行main.py将一键完成下述操作：

1. 载入数据集，划分训练、验证、测试集

2. 数据预处理：对训练数据标准化，并用相同的均值方差对验证和测试集标准化

3. 设定超参数取值并搜索最佳超参数组合，超参数包含

   nHidden : 隐藏层神经元数量

   lr : 初始学习率

   lam : L2正则化强度

4. 使用搜索给出的超参数组合进行训练

5. 对测试集预测并给出分类错误率

6. 保存模型的架构、网络参数

如果不进行搜索，在main.py中第35行

```python
# search
best_params = search(hyperparams, X, y_oh, Xvalid, yvalid, nLabels)
nHidden, lr, lam = best_params
# nHidden, lr, lam = [256], 1e-2, 1e-6
```

注释上方第2、3行并取消注释第4行，可以跳过搜索，使用给定超参数直接训练

#### 其他文件说明

- utils.py

  包含softmax、one-hot编码、标准化、网络前传预测等函数

- loss.py

  计算损失并回传梯度

- train.py

  给定超参数进行单次训练

- search.py

  在给定超参数的取值中搜索验证集错误率最小的超参数组合

- visualize.py

  将可视化训练和验证的loss曲线，验证的accuracy曲线，以及可视化每一层的网络参数

- npy文件

  保存网络结构、参数，训练过程的记录等
