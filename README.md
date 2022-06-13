# pytorch_chinese_QANet_cmrc2018
基于QANet的中文阅读理解。主要代码是基于：[heliumsea/QANet-pytorch: A PyTorch implementation of QANet. (github.com)](https://github.com/heliumsea/QANet-pytorch)，这里将其进行部分修改以适配中文的阅读理解，主要的还是去理解模型相关的一些组件以及如何将字向量和词向量进行融合。代码思想来源于该论文：[QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.09541)。使用的词向量来自这：[Embedding/Chinese-Word-Vectors: 100+ Chinese Word Vectors 上百种预训练中文词向量 (github.com)](https://github.com/Embedding/Chinese-Word-Vectors)，使用的是"Sogou News 搜狗新闻"的基于Word的300d，使用的字向量是进行numpy初始化的。数据及模型地址：<br>

链接：https://pan.baidu.com/s/1sYrL-iFsnywjcSePwrIy2g?pwd=lrsw <br>
提取码：lrsw<br>

# 目录结构

```
data：存储数据
log：tensorboard结果以及验证结果
model_hub：放置词向量
output：保存训练好的模型
config.py：参数配置
data_loader.py：数据加载
models.py：模型
preprocess.py：数据预处理
train.py：训练验证和测试
```

# 依赖

```
pytorch
tensorboardX
```

# 运行

```python
python train.py
```

# 结果

参数：

```python
import os


class config:
    target_dir = './data/'
    para_limit = 400  # 内容的最大长度
    ques_limit = 50  # 问题的额最大长度
    ans_limit = float("inf")  # 限制答案的最大长度
    char_limit = 10  # 限制token里面字符的最大长度
    word_emb_file = 'model_hub/sgns.sogou.word'  # 词向量
    char_emb_file = None  # 字向量

    word_emb_dim = 300  # 词向量维度
    char_emb_dim = 100  # 字向量维度
    train_record_file = os.path.join(target_dir, "train.npz")  # 训练特征保存路径
    dev_record_file = os.path.join(target_dir, "dev.npz")  # 验证特征保存路径
    save_word_emb_file = os.path.join(target_dir, "word_emb.pkl")  # 处理后词嵌入保存路径
    save_char_emb_file = os.path.join(target_dir, "char_emb.pkl")  #处理后字符嵌入保存路径
    train_eval_file = os.path.join(target_dir, "train_eval.json")  # 处理后寻来你样本路径
    dev_eval_file = os.path.join(target_dir, "dev_eval.json")  # 处理后验证样本路径
    dev_meta = os.path.join(target_dir, "dev_meta.json")  # 验证样本总数信息
    word2idx_file = os.path.join(target_dir, "word2idx.json")  # 词对应的idx
    char2idx_file = os.path.join(target_dir, "char2idx.json")  # 字符对应的idx
    save_dir = 'output/'  # 模型保存的位置

    mode = "train"
    num_epoch = 100
    learning_rate = 1e-3
    lr_warm_up_num = 0.01
    dropout_word = 0.3
    dropout_char = 0.3
    batch_size = 128
    val_num_batches = 128
    num_heads = 1
    connector_dim = 72
    pretrained_char = True  # 字向量是否可以被训练
    grad_clip = 5
    early_stop = 20
```

这里只运行了76个epoch就停止了，因为可能会过拟合：

![image](https://github.com/taishan1994/pytorch_chinese_QANet_cmrc2018/blob/main/image/result.png)

结果：best_em：35.28966300643695 best_f1：35.73951228627487

验证结果：

```
《战国无双3》是由哪两个公司合作开发的？
真实：光荣和ω-force
预测：光荣和ω-force
男女主角亦有专属声优这一模式是由谁改编的？
真实：村雨城
预测：任天堂游戏谜之村雨城改编
战国史模式主打哪两个模式？
真实：「战史演武」&「争霸演武」
预测：战史演武
锣鼓经是什么？
真实：大陆传统器乐及戏曲里面常用的打击乐记谱方法
预测：大陆传统器乐及戏曲里面常用的打击乐记谱方法
锣鼓经常用的节奏型称为什么？
真实：锣鼓点
预测：「锣鼓点
锣鼓经运用的程式是什么？
真实：依照角色行当的身份、性格、情绪以及环境，配合相应的锣鼓点。
预测：依照角色行当的身份、性格、情绪以及环境
戏曲锣鼓所运用的敲击乐器主要有什么类型？
真实：鼓、锣、钹和板
预测：鼓、锣、钹和板四类型
```

还是有一定效果的。

# 参考：

> [heliumsea/QANet-pytorch: A PyTorch implementation of QANet. (github.com)](https://github.com/heliumsea/QANet-pytorch)

