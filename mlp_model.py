import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionMLP(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attention_proj = nn.Linear(feat_dim * 2, feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
    def forward(self, feat1, feat2):
        if feat1.dim() == 1:
            feat1 = feat1.unsqueeze(0)
        if feat2.dim() == 1:
            feat2 = feat2.unsqueeze(0)
            
        combined = torch.cat([feat1, feat2], dim=-1)
        attention = torch.sigmoid(self.attention_proj(combined))
        
        weighted_feat1 = feat1 * attention
        weighted_feat2 = feat2 * (1 - attention)
        
        fused = self.mlp(torch.cat([weighted_feat1, weighted_feat2], dim=-1))
        return fused

class KAN_MoDTI(nn.Module):
    def __init__(self, num_drug, num_protein, enc_dim, MLP_depth,
                 affinity_threshold, device, QSOCTD_input_dim=103, max_seq_len=60):
        super(KAN_MoDTI, self).__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.enc_dim = enc_dim
        self.QSOCTD_input_dim = QSOCTD_input_dim
        
        self.drug_sequence_embed = nn.Embedding(num_drug, enc_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_len, enc_dim))
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim, 
            nhead=8, 
            dim_feedforward=512, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.node_transform = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, enc_dim)
        )
        
        self.edge_transform = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.ReLU(),
            nn.Linear(enc_dim // 2, enc_dim)
        )
        
        self.protein_embed = nn.Embedding(num_protein, enc_dim)
        
        self.qsoctd_hidden1 = nn.Linear(QSOCTD_input_dim, 128)
        self.qsoctd_hidden2 = nn.Linear(128, 64)
        # self.qsoctd_out = nn.Linear(64, 40)
        self.qsoctd_out = nn.Linear(64, enc_dim)
        self.qsoctd_dropout = nn.Dropout(0.3)
        
        self.drug_fusion = FeatureFusionMLP(enc_dim)
        self.protein_fusion = FeatureFusionMLP(enc_dim)
        
        self.mlp_model = nn.Sequential(
            nn.Linear(enc_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        self.dropout = nn.Dropout(0.3)
        self.threshold = affinity_threshold
        
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
    
    def process_graph_data(self, graph_data):
        if not isinstance(graph_data, dict):
            return torch.zeros(1, self.enc_dim, device=self.device)
        
        x = graph_data.get('x', torch.zeros(1, 1, device=self.device))
        edge_index = graph_data.get('edge_index', torch.zeros(2, 0, dtype=torch.long, device=self.device))
        edge_attr = graph_data.get('edge_attr', torch.zeros(0, 1, device=self.device))
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float, device=self.device)
        elif x.device != self.device:
            x = x.to(device=self.device)
            
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        elif edge_index.device != self.device:
            edge_index = edge_index.to(device=self.device)
            
        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=self.device)
        elif edge_attr.device != self.device:
            edge_attr = edge_attr.to(device=self.device)

        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.node_transform(x)
        
        try:
            x_mlp = self.mlp(x)
            graph_vector = torch.mean(x_mlp, dim=0, keepdim=True)
            return graph_vector
        except:
            return torch.mean(x, dim=0, keepdim=True)
    
    def qsoctd_encode(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = F.relu(self.qsoctd_hidden1(x))
        x = self.qsoctd_dropout(x)
        x = F.relu(self.qsoctd_hidden2(x))
        x = self.qsoctd_dropout(x)
        x = self.qsoctd_out(x)
        return x
    
    def process_single_sample(self, sample):
        if len(sample) < 4:
            return None, None
            
        protein_ngram, qsoctd_feature, molecule_word, graph_data = sample[:4]
        
        try:
            if isinstance(molecule_word, torch.Tensor):
                if molecule_word.device != self.device:
                    molecule_word = molecule_word.to(device=self.device, dtype=torch.long)
                else:
                    molecule_word = molecule_word.to(dtype=torch.long)
                    
            if molecule_word.dim() == 1:
                molecule_word = molecule_word.unsqueeze(0)
            
            molecule_word_embed = self.drug_sequence_embed(molecule_word)
            seq_len = min(molecule_word_embed.size(1), self.max_seq_len)
            molecule_word_embed = molecule_word_embed[:, :seq_len, :]
            pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0)
            molecule_word_embed = molecule_word_embed + pos_emb
            
            molecule_word_vector = self.transformer_encoder(molecule_word_embed)
            molecule_word_vector = torch.mean(molecule_word_vector, dim=1)
            
            compound_graph_vector = self.process_graph_data(graph_data)
            compound_vector = self.drug_fusion(molecule_word_vector, compound_graph_vector)
            
            if isinstance(protein_ngram, torch.Tensor):
                if protein_ngram.device != self.device:
                    protein_ngram = protein_ngram.to(device=self.device, dtype=torch.long)
                else:
                    protein_ngram = protein_ngram.to(dtype=torch.long)
                    
            if protein_ngram.dim() == 1:
                protein_ngram = protein_ngram.unsqueeze(0)
            
            protein_embed = self.protein_embed(protein_ngram)
            protein_embed = torch.mean(protein_embed, dim=1)
            
            if isinstance(qsoctd_feature, torch.Tensor):
                if qsoctd_feature.device != self.device:
                    qsoctd_feature = qsoctd_feature.to(device=self.device, dtype=torch.float)
                else:
                    qsoctd_feature = qsoctd_feature.to(dtype=torch.float)
                    
            if qsoctd_feature.dim() == 1:
                qsoctd_feature = qsoctd_feature.unsqueeze(0)
            
            qsoctd_encoded = self.qsoctd_encode(qsoctd_feature)
            protein_vector = self.protein_fusion(protein_embed, qsoctd_encoded)
            
            compound_protein = torch.cat([compound_vector, protein_vector], dim=1)
            return compound_protein, compound_vector
            
        except Exception as e:
            return None, None
    
    def forward(self, batch_data, train=True):
        # Start measuring time and memory before epoch starts
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0
        total_memory_change = 0  # Track the cumulative memory change

        if train:
            all_outputs = []
            batch_labels = []
            
            for sample in batch_data:
                try:
                    if len(sample) >= 5:
                        label = sample[4]
                        if isinstance(label, torch.Tensor):
                            label = label.item()
                        batch_labels.append(int(label))
                    else:
                        continue
                    
                    compound_protein, _ = self.process_single_sample(sample)
                    if compound_protein is None:
                        continue
                    
                    output = self.mlp_model(compound_protein)
                    all_outputs.append(output)
                except:
                    continue
            
            if not all_outputs:
                dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                return dummy_loss, dummy_loss, dummy_loss, dummy_loss
            
            all_outputs = torch.cat(all_outputs, dim=0)
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
            
            classification_loss = F.cross_entropy(all_outputs, labels_tensor)
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Measure elapsed time and memory usage after the entire epoch
            elapsed_time = time.time() - start_time
            end_memory = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0
            memory_diff = (end_memory - start_memory) / (1024 ** 2)  # Memory difference in MB
            total_memory_change += memory_diff
            
            print(f"Epoch time: {elapsed_time:.4f}s, Total memory change: {total_memory_change:.2f} MB")
            
            return classification_loss, dummy_loss, dummy_loss, dummy_loss
            
        else:
            all_labels = []
            all_predictions = []
            all_scores = []
            
            for sample in batch_data:
                try:
                    if len(sample) >= 5:
                        label = sample[4]
                        if isinstance(label, torch.Tensor):
                            label = label.item()
                        label = int(label)
                    else:
                        label = None
                    
                    compound_protein, _ = self.process_single_sample(sample)
                    if compound_protein is None:
                        continue
                    
                    output = self.mlp_model(compound_protein)
                    
                    prediction = torch.argmax(output, dim=1).item()
                    score = torch.softmax(output, dim=1)[0, 1].item()
                    
                    all_labels.append(label if label is not None else prediction)
                    all_predictions.append(prediction)
                    all_scores.append(score)
                except:
                    continue
            
            if not all_labels:
                return [0], [0], [0.5]
                
            return all_labels, all_predictions, all_scores
