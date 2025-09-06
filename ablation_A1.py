import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN

class FeatureFusionKAN(nn.Module):
    
    def __init__(self, feat_dim):
        super().__init__()
        self.attention_proj = nn.Linear(feat_dim * 2, feat_dim)
        self.kan = KAN([feat_dim * 2, feat_dim, feat_dim])

    def forward(self, feat1, feat2):
        if feat1.dim() == 1:
            feat1 = feat1.unsqueeze(0)
        if feat2.dim() == 1:
            feat2 = feat2.unsqueeze(0)

        combined = torch.cat([feat1, feat2], dim=-1)
        attention = torch.sigmoid(self.attention_proj(combined))

        weighted_feat1 = feat1 * attention
        weighted_feat2 = feat2 * (1 - attention)

        fused = self.kan(torch.cat([weighted_feat1, weighted_feat2], dim=-1))
        return fused


class KAN_MoDTI(nn.Module):
    
    def __init__(self,
                 num_drug,
                 num_protein,
                 enc_dim,
                 MLP_depth,
                 affinity_threshold,
                 device,
                 QSOCTD_input_dim=103,
                 max_seq_len=60):
        super(KAN_MoDTI, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.enc_dim = enc_dim
        self.threshold = affinity_threshold

        
        self.drug_sequence_embed = nn.Embedding(num_drug, enc_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_len, enc_dim))
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)

        
        self.protein_embed = nn.Embedding(num_protein, enc_dim)
        self.qsoctd_hidden1 = nn.Linear(QSOCTD_input_dim, 128)
        self.qsoctd_hidden2 = nn.Linear(128, 64)
        self.qsoctd_out = nn.Linear(64, 40)
        self.qsoctd_dropout = nn.Dropout(0.3)
        self.protein_fusion = FeatureFusionKAN(enc_dim)

        
        self.KAN_model = KAN([enc_dim * 2, 64, 2])
        self.dropout = nn.Dropout(0.3)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def qsoctd_encode(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.qsoctd_hidden1(x))
        x = self.qsoctd_dropout(x)
        x = F.relu(self.qsoctd_hidden2(x))
        x = self.qsoctd_dropout(x)
        return self.qsoctd_out(x)

    def process_single_sample(self, sample):
        
        if len(sample) < 4:
            return None, None
        protein_ngram, qsoctd_feature, molecule_word, _ = sample[:4]

        
        molecule_word = molecule_word.to(device=self.device, dtype=torch.long)
        if molecule_word.dim() == 1:
            molecule_word = molecule_word.unsqueeze(0)
        emb = self.drug_sequence_embed(molecule_word)
        seq_len = min(emb.size(1), self.max_seq_len)
        emb = emb[:, :seq_len, :] + self.pos_embedding[:seq_len, :].unsqueeze(0)
        seq_encoded = self.transformer_encoder(emb)
        drug_vec = torch.mean(seq_encoded, dim=1)

        
        protein_ngram = protein_ngram.to(device=self.device, dtype=torch.long)
        if protein_ngram.dim() == 1:
            protein_ngram = protein_ngram.unsqueeze(0)
        prot_emb = self.protein_embed(protein_ngram)
        prot_emb = torch.mean(prot_emb, dim=1)

        qsoctd_feature = qsoctd_feature.to(device=self.device, dtype=torch.float)
        if qsoctd_feature.dim() == 1:
            qsoctd_feature = qsoctd_feature.unsqueeze(0)
        qso_vec = self.qsoctd_encode(qsoctd_feature)

        prot_vec = self.protein_fusion(prot_emb, qso_vec)

        
        pair_vec = torch.cat([drug_vec, prot_vec], dim=1)
        return pair_vec, drug_vec

    def forward(self, batch_data, train=True):
        if train:
            all_outputs, batch_labels = [], []
            for sample in batch_data:
                if len(sample) < 5:
                    continue
                label = sample[4]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                batch_labels.append(int(label))

                pair_vec, _ = self.process_single_sample(sample)
                if pair_vec is None:
                    continue
                output = self.KAN_model(pair_vec)
                all_outputs.append(output)

            if not all_outputs:
                dummy = torch.tensor(0.0, device=self.device, requires_grad=True)
                return dummy, dummy, dummy, dummy

            outputs = torch.cat(all_outputs, dim=0)
            labels = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
            classification_loss = F.cross_entropy(outputs, labels)
            dummy = torch.tensor(0.0, device=self.device, requires_grad=True)
            return classification_loss, dummy, dummy, dummy

        else:
            all_labels, all_predictions, all_scores = [], [], []
            for sample in batch_data:
                pair_vec, _ = self.process_single_sample(sample)
                if pair_vec is None:
                    continue
                output = self.KAN_model(pair_vec)
                pred = torch.argmax(output, dim=1).item()
                score = torch.softmax(output, dim=1)[0, 1].item()
                all_predictions.append(pred)
                all_scores.append(score)

                label = sample[4] if len(sample) >= 5 else pred
                if isinstance(label, torch.Tensor):
                    label = label.item()
                all_labels.append(int(label))

            return all_labels, all_predictions, all_scores
