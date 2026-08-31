#!/usr/bin/env python3
"""GloVe 原格式 → gensim word2vec 格式（gensim 4 已删除官方 glove2word2vec，手动加表头即可）
用法: python convert_glove.py glove.840B.300d.txt glove.840B.300d.gensim.txt
"""
import sys

def convert(src, dst, dim=300):
    with open(src, encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)   # glove.840B.300d 应为 2196017
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        g.write(f"{n_lines} {dim}\n")  # gensim word2vec 格式要求首行声明「词数 维度」
        for line in f:
            g.write(line)
    print(f"done: {n_lines} words x {dim}d -> {dst}")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
