# 利用Bert进行fine-tune，用于NER任务


# What's New

## Updated June 21, 2022
* 修复了几个bug
* 增加了对文件进行推理

## Updated June 17, 2022
* 增加了在bert顶层添加crf层的设计，在给定数据集上实现了average f1  1个点的提升，可通过修改 ./conf/train.yaml 中 use_crf 进行设定
* 推理时增加了批预测的情形，考虑到推理文本量，实现时未采用批计算的方法，会对推理速度产生影响，后续可考虑实现批计算


# 快速开始

### 环境依赖

> python == 3.8 

- pytorch-transformers == 1.2.0
- torch == 1.5.0
- hydra-core == 1.0.6
- seqeval == 1.2.2
- tqdm == 4.60.0
- matplotlib == 3.4.1


### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖：`pip install -r requirements.txt`


### 使用数据进行训练预测

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/ner/standard/data.tar.gz```在此目录下

  在`data`文件夹下存放数据：
  
  - `train.txt`：存放训练数据集
  - `valid.txt`：存放验证数据集
  - `test.txt`：存放测试数据集
- 开始训练：```python train.py``` (训练所用到参数都在conf文件夹中，修改即可)

- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存在 `checkpoints` 文件夹内。

- 进行预测 ```python predict.py```


### 模型内容

BERT

[CRF](https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/)


### TODO
* 推理部分实现批处理


### 参考
https://github.com/zjunlp/DeepKE


