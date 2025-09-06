


import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from collections import defaultdict
from tqdm import tqdm
from rdkit.Chem import AllChem


CHAR_SMI_SET = {
    "(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
    "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
    "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
    "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
    "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
    "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
    "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
    "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64
}


class Smile2Morgan:
    def __init__(self, shape=1024, radius=2):
        self.shape = shape
        self.radius = radius

    def canonicalize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return None

    def smiles_to_morgan(self, smile: str):
        smile = self.canonicalize(smile)
        mol = Chem.MolFromSmiles(smile)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.shape)
        features = np.zeros((self.shape,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features


def QSO_CTD_feature(fasta_file, data_path, maxlag, output_name):
    
    try:
        if os.path.getsize(fasta_file) == 0:
            write_fasta = True
        else:
            write_fasta = False
    except FileNotFoundError:
        write_fasta = True

    
    filtered_protein_list = []
    filtered_data_lines = []
    with open(data_path, 'r') as f:
        lines = f.read().strip().split('\n')
    if 'Drugbank' in data_path:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            seq = parts[3]
            if len(seq) >= maxlag + 1:
                filtered_protein_list.append(seq)
                filtered_data_lines.append(line)
            else:
                print("Sequence '{}' is too short (length {}), skipping.".format(parts[0], len(seq)))
    else:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            seq = parts[1]
            if len(seq) >= maxlag + 1:
                filtered_protein_list.append(seq)
                filtered_data_lines.append(line)
            else:
                print("Sequence '{}' is too short (length {}), skipping.".format(parts[0], len(seq)))
    
    
    filtered_data_path = os.path.splitext(data_path)[0] + "_filtered.txt"
    with open(filtered_data_path, "w") as f:
        for line in filtered_data_lines:
            f.write(line + "\n")
    print("Filtered data written to: {}".format(filtered_data_path))
    
    
    if write_fasta:
        print(f'Writing fasta file to: {fasta_file}')
        with open(fasta_file, "w") as fasta:
            for idx, protein in enumerate(filtered_protein_list):
                fasta.write(f'>Protein{idx}\n')
                fasta.write(f'{protein}\n')
        print('Fasta writing finished.')
    else:
        print("FASTA file already exists. Consider regenerating if needed.")

    print('Starting to process QSO and CTD features, this may take minutes even hours to finish...')
    if not os.path.exists(output_name):
        
        from iFeature.codes.QSOrder import QSOrder, readFasta as readFastaQSOrder, saveCode as saveCodeQSOrder
        from iFeature.codes.CTDCClass import CTDCClass, readFasta as readFastaCTDC, saveCode as saveCodeCTDC
        
        fastas = readFastaQSOrder.readFasta(fasta_file)
        
        
        qso_encodings = QSOrder(fastas, maxlag=maxlag)
        qso_file = output_name.replace('.tsv', '_QSO.tsv')
        saveCodeQSOrder.savetsv(qso_encodings, qso_file)
        
        
        groups = ['RKEDQN', 'GASTPHY', 'CLVIMFW']
        ctd_encodings = CTDCClass(fastas, groups)
        ctd_file = output_name.replace('.tsv', '_CTD.tsv')
        saveCodeCTDC.savetsv(ctd_encodings, ctd_file)
        
        
        qso_df = pd.read_csv(qso_file, sep='\t', header=None)
        ctd_df = pd.read_csv(ctd_file, sep='\t', header=None)
        combined_features = pd.concat([qso_df.iloc[:, 1:], ctd_df.iloc[:, 1:]], axis=1)
        combined_features.insert(0, 0, qso_df.iloc[:, 0])
        combined_features.to_csv(output_name, sep='\t', index=False, header=False)
        print('QSO and CTD features processing finished.')
    elif os.path.getsize(output_name) > 4096:
        print('QSO and CTD features already processed.')


def smiles_to_graph(smile):
    
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([atom.GetAtomicNum()])
    node_features = np.array(node_features)

    
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        bond_type = bond.GetBondTypeAsDouble()
        edge_attr.append([bond_type])
        edge_attr.append([bond_type])
    if len(edge_index) > 0:
        edge_index = np.array(edge_index).T
        edge_attr = np.array(edge_attr)
    else:
        edge_index = np.empty((2, 0))
        edge_attr = np.empty((0, 1))
    
    return {'x': node_features, 'edge_index': edge_index, 'edge_attr': edge_attr}


class process():
    def __init__(self, data_path):
        self.wordDict={}
        
        with open(data_path, 'r') as f:
            dataList = f.read().strip().split('\n')
            
            dataList = [d for d in dataList if '.' not in d.strip().split()[0]]
        print(f"There are {len(dataList)} pairs in filtered data.")

        
        encodings = pd.read_csv(os.path.join(os.path.dirname(data_path), 'QSOCTD.tsv'),
                                sep='\t', header=None)
        try:
            if len(encodings) <= 1:
                raise ValueError("The length of encodings is too small, check your fasta file.")
            print("Encodings length is sufficient")
        except ValueError as e:
            print(f"Error: {e}")

        features_list = []
        for number, data in tqdm(enumerate(dataList), total=len(dataList), desc="Processing features"):
            
            if 'Drugbank' in data_path:
                _, _, smile, sequence, interaction = data.strip().split()
            else:
                smile, sequence, interaction = data.strip().split()

            # 药物侧 SMILES 数字序列
            molecule_word = np.array([CHAR_SMI_SET[ch] for ch in smile])
            # 药物侧图数据对象
            graph_data = smiles_to_graph(smile)
            # 蛋白质侧 QSO+CTD 特征：从全局特征文件中加载对应特征（假设第一行为 header，从第二行开始读取）
            QSOCTD_feature_ = np.array(encodings.iloc[number + 1][1:]).astype(np.float32)
            # 蛋白质侧 n-gram 特征（利用原始蛋白质序列生成 n-gram 映射，这里 ngram=3，可根据需要修改）
            protein_ngram = self.splitSeqeunce(sequence, ngram=3)
            # 标签（例如相互作用值）
            label = np.array([float(interaction)])

            # 最终元组： (蛋白质 n-gram 特征, 蛋白质 QSO+CTD 特征, 药物 SMILES 序列特征, 药物图特征, 标签)
            features_tuple = (
                
                np.array(protein_ngram, dtype=np.int32),
                np.array(QSOCTD_feature_, dtype=np.float32),
                np.array(molecule_word, dtype=np.int32),
                graph_data,
                np.array(label, dtype=np.float32)
            )
            features_list.append(features_tuple)

        path = os.path.dirname(data_path)
        np.save(path + '/features.npy', np.array(features_list, dtype=object), allow_pickle=True)
        
        with open(path + '/smiles_mapping.pickle', 'wb') as f:
            pickle.dump(dict(CHAR_SMI_SET), f)
        with open(path + '/wordDict.pickle', 'wb') as f:
            pickle.dump(self.wordDict, f)
        print('All feature file writings finished.')

    def splitSeqeunce(self, sequence, ngram):
        
        sequence = '-' + sequence + '='
        words = [sequence[i:i + ngram] for i in range(len(sequence) - ngram + 1)]
        return np.array(words)

    def splitSeqeunce(self, sequence, ngram):
    
        sequence = '-' + sequence + '='
        words = []
    
        for i in range(len(sequence) - ngram + 1):
            ngram_seq = sequence[i:i + ngram]
            if ngram_seq not in self.wordDict:
            
                self.wordDict[ngram_seq] = len(self.wordDict)
            words.append(self.wordDict[ngram_seq])
    
        return np.array(words, dtype=np.int32)


# 主程序
if __name__ == '__main__':
    data_path = 'ki'
    fasta_file = 'ki/data_ki_protein_sequences.fasta'
    # 生成蛋白质 QSO 和 CTD 特征（保持蛋白质侧特征不变）
    QSO_CTD_feature(fasta_file, data_path=data_path + '/bindingdb.txt',
                    maxlag=30, output_name=data_path + '/QSOCTD.tsv')
    # process() 使用过滤后的数据文件，生成的 features.npy 包含蛋白质 n-gram、QSO+CTD、药物 SMILES 序列、药物图特征和标签
    process(os.path.splitext(data_path + '/bindingdb.txt')[0] + "_filtered.txt")
