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

