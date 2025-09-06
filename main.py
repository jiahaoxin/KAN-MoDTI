import itertools
import os
import random
import time
import torch
import pickle
import argparse
import numpy as np
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from CustomDataset import CustomDataset
import sys
from model import KAN_MoDTI


# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--ld", type=float, default=0.5)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--features", type=int, default=40)
parser.add_argument("--MLP_depth", type=int, default=3)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument("--dataset", type=str, default='human')
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
parser.add_argument('--max_seq_len', type=int, default=60)
parser.add_argument('--verbose', type=bool, default=False)

args = parser.parse_args()

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置随机种子
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 进度条参数
term_width = 80
TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()
    
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    
    L = []
    if msg:
        L.append(' | ' + msg)
    
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')
    
    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))
    
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def loadPickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def get_batch(batch_index, pos_list, neg_list, batch_size):
    half = batch_size // 2
    start = batch_index
    end = min(start + half, len(pos_list))
    pos_batch = []
    for i in range(start, end):
        sample = list(pos_list[i])
        while len(sample) < 5: sample.append(None)
        sample[4] = 1
        pos_batch.append(sample)
    neg_batch = []
    neg_batch_size = batch_size - len(pos_batch)
    neg_start = batch_index % len(neg_list)
    neg_end = min(neg_start + neg_batch_size, len(neg_list))
    for i in range(neg_start, neg_end):
        sample = list(neg_list[i])
        while len(sample) < 5: sample.append(None)
        sample[4] = 0
        neg_batch.append(sample)
    # 补齐
    if len(neg_batch) < neg_batch_size:
        need = neg_batch_size - len(neg_batch)
        for i in range(min(need, len(neg_list))):
            sample = list(neg_list[i])
            while len(sample) < 5: sample.append(None)
            sample[4] = 0
            neg_batch.append(sample)
    batch = pos_batch + neg_batch
    if batch:
        random.shuffle(batch)
    return batch


def train(args, model, train_data, optimizer, scheduler):
    model.train()
    pos_list, neg_list = train_data
    half = args.batch_size // 2
    total_batches = min(len(pos_list)//half, len(neg_list)//half)
    total_loss = 0
    total_class_loss = 0
    count = 0
    for i in range(0, len(pos_list), half):
        optimizer.zero_grad()
        batch = get_batch(i, pos_list, neg_list, args.batch_size)
        if not batch: continue
        cls_loss, _, _, _ = model(batch)
        if torch.isnan(cls_loss): continue
        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_class_loss += cls_loss.item()
        total_loss += cls_loss.item()
        count += 1
        progress_bar(count-1, total_batches, f'Loss: {cls_loss.item():.3f}')
    if count>0:
        avg = total_loss/count
        scheduler.step(avg)
        print(f"Classification Loss: {total_class_loss/count:.4f}")
        print(f"Total Loss: {avg:.4f}")
        return avg
    return float('inf')

def test(args, model, test_data):
    model.eval()
    pos_list, neg_list = test_data
    all_labels, all_preds, all_scores = [], [], []
    half = args.batch_size // 2
    with torch.no_grad():
        for i in range(0, len(pos_list), half):
            batch = get_batch(i, pos_list, neg_list, args.batch_size)
            if not batch: continue
            labels, preds, scores = model(batch, train=False)
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_scores.extend(scores)
    if not all_labels:
        return [np.array([0])], [np.array([0])], [np.array([0.5])]
    return [np.array(all_labels)], [np.array(all_preds)], [np.array(all_scores)]


def main():
    # 创建检查点目录
    os.makedirs(args.checkpoint_path, exist_ok=True)
    
    # —— 加载并解包训练/测试数据 —— 
    data_path = args.dataset
    train_pos, train_neg = CustomDataset(os.path.join(data_path, 'features.npy')).get_train_data()
    test_pos,  test_neg  = CustomDataset(os.path.join(data_path, 'features.npy')).get_test_data()
    train_data = (train_pos, train_neg)
    test_data  = (test_pos,  test_neg)
    
    _, qsoctd_feat, _, _, _ = train_pos[0]
    # 如果 QSOCTD_feature 是一维向量：
    QSOCTD_input_dim = qsoctd_feat.shape[-1]

   
    compound_len = len(loadPickle(os.path.join(data_path, 'smiles_mapping.pickle')))
    protein_len = len(loadPickle(os.path.join(data_path, 'wordDict.pickle')))
    
    print(f"Data dimensions:")
    print(f"QSOCTD input dimension: {QSOCTD_input_dim}")
    print(f"Compound dictionary size: {compound_len}")
    print(f"Protein dictionary size: {protein_len}")
    
    # 初始化模型
    model = KAN_MoDTI(
        num_drug=compound_len,
        num_protein=protein_len,
        enc_dim=args.features,
        MLP_depth=args.MLP_depth,
        affinity_threshold=None,
        device=device,
        QSOCTD_input_dim=103,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    # 加载检查点（如果需要）
    if args.resume:
        checkpoint_file = os.path.join(args.checkpoint_path, f'{args.dataset}_latest.pth')
        if os.path.exists(checkpoint_file):
            model.load_state_dict(torch.load(checkpoint_file))
            print(f"Loaded checkpoint: {checkpoint_file}")
    
    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.ld, patience=3, verbose=args.verbose
    )
    
    # 训练循环
    best_metrics = {'auc': 0, 'aupr': 0, 'precision': 0, 'recall': 0}
    metrics_history = {'auc': [], 'aupr': [], 'precision': [], 'recall': []}
    
    print(f"\nStarting training...")
    print(f"Total epochs: {args.epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"dataset: {args.dataset}")
    
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epoch}")
        print("-" * 60)
        
        # 训练
        train_loss = train(args, model, train_data, optimizer, scheduler)
        
        # 测试
        labels, predictions, scores = test(args, model, test_data)
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        scores = np.concatenate(scores)
        
        try:
            metrics = {
                'auc': roc_auc_score(labels, scores),
                'precision': precision_score(labels, predictions),
                'recall': recall_score(labels, predictions)
            }
            precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
            metrics['aupr'] = auc(recall_curve, precision_curve)
            
            for key in best_metrics:
                if metrics[key] > best_metrics[key]:
                    best_metrics[key] = metrics[key]
                metrics_history[key].append(metrics[key])
            
            # 打印当前epoch结果
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"AUC: {metrics['auc']:.6f}")
            print(f"AUPR: {metrics['aupr']:.6f}")
            print(f"Precision: {metrics['precision']:.6f}")
            print(f"Recall: {metrics['recall']:.6f}")
            print(f"Time: {time.time()-epoch_start_time:.2f}s")
            
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(args.checkpoint_path, 
                                       f'{args.dataset}_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")
            
            latest_path = os.path.join(args.checkpoint_path, f'{args.dataset}_latest.pth')
            torch.save(model.state_dict(), latest_path)
            
        except Exception as e:
            print(f"Error in metric calculation: {e}")
            continue
    

    print("\nTraining completed!")
    print("\nBest metrics:")
    for key, value in best_metrics.items():
        print(f"{key.upper()}: {value:.6f}")
    
    final_path = os.path.join(args.checkpoint_path, f'{args.dataset}_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model to {final_path}")

if __name__ == '__main__':
    main()
