import random
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path, cold_split_ratio=0.1, seed=0):
        # ———— 固定随机种子 ————
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            self.device = torch.device('cpu')

        # ———— 1. 载入所有样本 ————
        all_samples = self._load_npy(file_path)

        # ———— 2. 提取所有蛋白质和分子 ID ————
        prot_ids = set()
        mol_ids  = set()
        for prot, qso, mol, graph, label in all_samples:
            pid = tuple(prot.cpu().numpy().tolist())
            mid = tuple(mol.cpu().numpy().tolist())
            prot_ids.add(pid)
            mol_ids.add(mid)

        prot_ids = list(prot_ids)
        mol_ids  = list(mol_ids)
        random.shuffle(prot_ids)
        random.shuffle(mol_ids)

        # ———— 3. 划分蛋白质 & 分子 ————
        n_test_prot = int(len(prot_ids) * cold_split_ratio)
        n_test_mol  = int(len(mol_ids)  * cold_split_ratio)

        test_prots = set(prot_ids[:n_test_prot])
        train_prots = set(prot_ids[n_test_prot:])
        test_mols  = set(mol_ids[:n_test_mol])
        train_mols  = set(mol_ids[n_test_mol:])

        # ———— 4. 按双冷条件分配样本 ————
        train_samples = []
        test_samples  = []
        for sample in all_samples:
            prot, qso, mol, graph, label = sample
            pid = tuple(prot.cpu().numpy().tolist())
            mid = tuple(mol.cpu().numpy().tolist())

            if (pid in train_prots) and (mid in train_mols):
                train_samples.append(sample)
            elif (pid in test_prots) and (mid in test_mols):
                test_samples.append(sample)
            # 其余样本（只满足单一冷）不用于训练和测试，以保证严格的双冷评估

        # ———— 5. 训练集正负平衡 ————
        pos_train = [s for s in train_samples if s[4].item() == 1.0]
        neg_train = [s for s in train_samples if s[4].item() == 0.0]
        if len(pos_train) > len(neg_train):
            neg_train += random.choices(neg_train, k=len(pos_train) - len(neg_train))
        else:
            pos_train += random.choices(pos_train, k=len(neg_train) - len(pos_train))

        # ———— 6. 保存分割结果 ————
        self.train_pos = pos_train
        self.train_neg = neg_train
        self.test_pos  = [s for s in test_samples if s[4].item() == 1.0]
        self.test_neg  = [s for s in test_samples if s[4].item() == 0.0]
        self.train_len = len(self.train_pos)

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        pos = self.train_pos[idx]
        neg = self.train_neg[idx]
        return self._pack(pos), self._pack(neg)

    def _pack(self, sample):
        inputs = sample[:-1]
        label  = sample[-1]
        return inputs, label

    def get_train_data(self):
        return self.train_pos, self.train_neg

    def get_test_data(self):
        return self.test_pos, self.test_neg

    def _load_npy(self, file_name):
        raw = np.load(file_name, allow_pickle=True)
        feats = []
        for sample in raw:
            prot_ngram = torch.LongTensor(sample[0]).to(self.device)
            qso_feat   = torch.FloatTensor(sample[1]).to(self.device)
            mol_word   = torch.LongTensor(sample[2]).to(self.device)
            graph_data = sample[3]
            label      = torch.FloatTensor([sample[4]]).to(self.device)
            feats.append((prot_ngram, qso_feat, mol_word, graph_data, label))
        return feats
