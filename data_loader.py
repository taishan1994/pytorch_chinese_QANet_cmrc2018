import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class SQuADDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.context_idxs = data["context_idxs"]
        self.context_char_idxs = data["context_char_idxs"]
        self.ques_idxs = data["ques_idxs"]
        self.ques_char_idxs = data["ques_char_idxs"]
        self.y1s = data["y1s"]
        self.y2s = data["y2s"]
        self.ids = data["ids"]
        self.num = len(self.ids)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.context_idxs[idx], self.context_char_idxs[idx], self.ques_idxs[idx], self.ques_char_idxs[idx], \
               self.y1s[idx], self.y2s[idx], self.ids[idx]


def collate(data):
    Cwid, Ccid, Qwid, Qcid, y1, y2, ids = zip(*data)
    Cwid = torch.tensor(Cwid).long()
    Ccid = torch.tensor(Ccid).long()
    Qwid = torch.tensor(Qwid).long()
    Qcid = torch.tensor(Qcid).long()
    y1 = torch.from_numpy(np.array(y1)).long()
    y2 = torch.from_numpy(np.array(y2)).long()
    ids = torch.from_numpy(np.array(ids)).long()
    return Cwid, Ccid, Qwid, Qcid, y1, y2, ids


def get_loader(npz_file, batch_size, mode="train"):
    dataset = SQuADDataset(npz_file)
    if mode == "train":
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 collate_fn=collate)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 collate_fn=collate)
    return data_loader


if __name__ == '__main__':
    from config import config
    npz_file = config.train_record_file
    # train_dataset = SQuADDataset(npz_file)
    # print(train_dataset[0])
    train_loader = get_loader(npz_file, batch_size=16, mode="train")
    for i,batch in enumerate(train_loader):
        print(batch[0].shape)
        break