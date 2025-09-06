import random
import numpy as np
import torch
from torch.utils.data import Dataset, random_split


seed = 0
torch.manual_seed(seed)
random.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
	def __init__(self, file_path, regression_split_threshold=None):
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.pos_sample, self.neg_sample = [], []
		self.features = self.loadNpy(file_path)
		
		for i, f in enumerate(self.features):
			if f[4] == 1:
				self.pos_sample.append(f)
			else:
				self.neg_sample.append(f)
		
			
		if len(self.pos_sample) > len(self.neg_sample):
			pos_neg_difference = len(self.pos_sample) - len(self.neg_sample)
			sampled_elements = random.sample(self.neg_sample, pos_neg_difference)
			self.neg_sample.extend(sampled_elements)
		elif len(self.pos_sample) < len(self.neg_sample):
			pos_neg_difference = len(self.neg_sample) - len(self.pos_sample)
			sampled_elements = random.sample(self.pos_sample, pos_neg_difference)
			self.pos_sample.extend(sampled_elements)
	
		train_size = int(0.8 * len(self.pos_sample))
		test_size = len(self.pos_sample) - train_size
		
		self.pos_train_data, self.pos_test_data = random_split(self.pos_sample, [train_size, test_size])
		self.neg_train_data, self.neg_test_data = random_split(self.neg_sample, [train_size, test_size])
		
	def __len__(self):
		return len(self.pos_train_data)
	
	def __getitem__(self, idx):
		pos_sample = self.pos_train_data[idx]
		neg_sample = self.neg_train_data[idx]
		
		pos_input, pos_label = pos_sample[:-1], pos_sample[-1]
		neg_input, neg_label = neg_sample[:-1], neg_sample[-1]
		
		return (pos_input, pos_label), (neg_input, neg_label)
	
	def get_train_data(self):
		return self.pos_train_data, self.neg_train_data
	
	def get_test_data(self):
		return self.pos_test_data, self.neg_test_data
	
	def loadNpy(self, fileName):
		
		data = np.load(fileName, allow_pickle=True)
		features =[]
		for sample in data:
        # 假设 sample = (protein_ngram, QSOCTD_feature, molecule_word, graph_data, label)
			protein_ngram = torch.LongTensor(sample[0]).to(self.device)
			QSOCTD_feature = torch.FloatTensor(sample[1]).to(self.device)
			molecule_word = torch.LongTensor(sample[2]).to(self.device)
			graph_data = sample[3]  # 保持原来的图数据字典格式
			label = torch.FloatTensor(sample[4]).to(self.device)
			features.append((protein_ngram, QSOCTD_feature, molecule_word, graph_data, label))
		return features
		
	
	def get_visualize_data(self):
		return self.features
	
