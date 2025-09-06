

import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="统计 TXT 格式配体–靶标数据集概览")
    parser.add_argument(
        "infile",
        help="输入的 txt 文件路径，格式：SMILES<空格>ProteinSeq<空格>Label（0/1）"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) 先读入，不给 label 指定 dtype
    df = pd.read_csv(
        args.infile,
        sep=r"\s+",
        header=None,
        names=["smiles", "sequence", "label"],
        dtype={"smiles": str, "sequence": str},
        on_bad_lines="skip",     # 遇到不规范行直接跳过
        engine="python"          # python 引擎对复杂分隔更健壮
    )

    # 2) 丢弃任何列中有缺失的行
    df.dropna(subset=["smiles", "sequence", "label"], inplace=True)

    # 3) 强制将 label 转为数字，然后去除无法转换的
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])       # 丢弃无法转为数值的
    df["label"] = df["label"].astype(int)  # 现在安全地转成 int

    # 4) 统计
    total_pairs  = len(df)
    unique_mols  = df["smiles"].nunique()
    unique_prots = df["sequence"].nunique()
    positives    = int(df["label"].sum())
    negatives    = total_pairs - positives

    # 5) 输出结果
    print("===== 数据集统计结果 =====")
    print(f"总配对数       : {total_pairs}")
    print(f"唯一药物分子数 : {unique_mols}")
    print(f"唯一蛋白质数   : {unique_prots}")
    print(f"阳性样本数     : {positives}")
    print(f"阴性样本数     : {negatives}")

if __name__ == "__main__":
    main()
