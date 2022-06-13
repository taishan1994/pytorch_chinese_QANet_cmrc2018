import math
import os
import numpy as np
import json
import re
from collections import Counter
import string
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from tensorboardX import SummaryWriter

from config import config
from data_loader import get_loader
from preprocess import preproc

writer = SummaryWriter(log_dir='./log')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        # exclude.update('，', '。', '、', '；', '「', '」')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def train(model, optimizer, scheduler, dataset, start):
    model.train()
    losses = []
    print(f'Training epoch {start}')
    for i, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(dataset):
        optimizer.zero_grad()
        """
        torch.Size([16, 400])
        torch.Size([16, 400, 10])
        torch.Size([16, 50])
        torch.Size([16, 50, 10])
        """
        Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        y1, y2 = y1.to(device), y2.to(device)
        p1 = F.log_softmax(p1, dim=1)
        p2 = F.log_softmax(p2, dim=1)
        loss1 = F.nll_loss(p1, y1)
        loss2 = F.nll_loss(p2, y2)
        loss = (loss1 + loss2)
        writer.add_scalar('data/loss', loss.item(), i + start * len(dataset))
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        print("\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()), end='')


def test(model, dataset, eval_file, test_i):
    print("\nTest")
    model.eval()
    answer_dict = {}
    losses = []
    num_batches = config.val_num_batches
    with torch.no_grad():
        for i, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(dataset):
            Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)

            P1, P2 = model(Cwid, Ccid, Qwid, Qcid)
            y1, y2 = y1.to(device), y2.to(device)
            p1 = F.log_softmax(P1, dim=1)
            p2 = F.log_softmax(P2, dim=1)
            loss1 = F.nll_loss(p1, y1)
            loss2 = F.nll_loss(p2, y2)
            loss = torch.mean(loss1 + loss2)
            losses.append(loss.item())

            p1 = F.softmax(P1, dim=1)
            p2 = F.softmax(P2, dim=1)

            # ymin = []
            # ymax = []
            outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
            for j in range(outer.size()[0]):
                outer[j] = torch.triu(outer[j])
                # outer[j] = torch.tril(outer[j], config.ans_limit)
            a1, _ = torch.max(outer, dim=2)
            a2, _ = torch.max(outer, dim=1)
            ymin = torch.argmax(a1, dim=1)
            ymax = torch.argmax(a2, dim=1)

            answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), ymin.tolist(), ymax.tolist())
            answer_dict.update(answer_dict_)
            print("\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()), end='')
            if ((i + 1) == num_batches):
                break
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    f = open("log/answers.json", "w", encoding='utf-8')
    json.dump(answer_dict, f, ensure_ascii=False)
    f.close()
    metrics["loss"] = loss
    print("EVAL loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"], metrics["exact_match"]))
    if config.mode == "train":
        writer.add_scalar('data/test_loss', loss, test_i)
        writer.add_scalar('data/F1', metrics["f1"], test_i)
        writer.add_scalar('data/EM', metrics["exact_match"], test_i)
    return metrics


def train_entry(config):
    from models import QANet

    with open(config.save_word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.save_char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r", encoding='utf-8') as fh:
        dev_eval_file = json.load(fh)

    print("Building model...")

    train_dataset = get_loader(config.train_record_file, config.batch_size, mode="train")
    dev_dataset = get_loader(config.dev_record_file, config.batch_size, mode="dev")

    lr = config.learning_rate
    base_lr = 1
    lr_warm_up_num = config.lr_warm_up_num

    model = QANet(word_mat, char_mat).to(device)


    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(lr=base_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=5e-8, params=parameters)
    cr = lr / math.log2(lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < lr_warm_up_num else lr)
    best_f1 = 0
    best_em = 0
    patience = 0
    unused = False

    for iter in range(config.num_epoch):
        train(model, optimizer, scheduler, train_dataset, iter)


        metrics = test(model, dev_dataset, dev_eval_file, (iter + 1) * len(train_dataset))
        dev_f1 = metrics["f1"]
        dev_em = metrics["exact_match"]
        if dev_f1 < best_f1 and dev_em < best_em:
            patience += 1
            if patience > config.early_stop:
                break
        else:
            patience = 0
            best_f1 = max(best_f1, dev_f1)
            best_em = max(best_em, dev_em)
            print('best_em：{} best_f1：{}'.format(best_em, best_f1))
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'model.pt'))

        fn = os.path.join(config.save_dir, "model.pt")
        torch.save(model, fn)


def test_entry(config):
    with open(config.dev_eval_file, "r", encoding='utf-8') as fh:
        dev_eval_file = json.load(fh)
    dev_dataset = get_loader(config.dev_record_file, config.batch_size)
    fn = os.path.join(config.save_dir, "model.pt")
    model = torch.load(fn)
    test(model, dev_dataset, dev_eval_file, 0)


def main(config):
    if config.mode == "train":
        train_entry(config)
    elif config.mode == "data":
        preproc(config)
    elif config.mode == "debug":
        config.batch_size = 2
        config.num_steps = 32
        config.val_num_batches = 2
        config.checkpoint = 2
        config.period = 1
        train_entry(config)
    elif config.mode == "test":
        test_entry(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == '__main__':
    main(config)
