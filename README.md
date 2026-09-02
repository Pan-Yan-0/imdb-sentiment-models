# IMDB 情感分类 · 深度学习模型集

文本分类技术演进线的后半段：从 TextCNN / RNN 系一路到 BERT 系。
配套前置仓库：[word2vec-nlp-tutorial](https://github.com/Pan-Yan-0/word2vec-nlp-tutorial)（词袋 / Word2Vec 阶段，分数 0.846 / 0.844）。

> **运行环境：Kaggle Notebook**（免费 29GB 内存 + T4×2/P100 GPU 30h/周）。GloVe 840B 加载需 6-8GB 内存，普通笔记本装不下，全程在 Kaggle 上跑。

## 结构

| 组 | 文件 | 说明 |
|---|---|---|
| 预处理 | `imdb_process.py` | 清洗 + 建词表 + 加载 GloVe → 生成 `pickle/imdb_glove.pickle3`（先跑且只跑一次） |
| GloVe 系 | `imdb_cnn.py` `imdb_lstm.py` `imdb_gru.py` `imdb_attention_lstm.py` `imdb_cnnlstm.py` `imdb_capsule_lstm.py` `imdb_transformer.py` | 静态词向量 + PyTorch 手写网络，依赖 pickle |
| BERT 系 | `imdb_bert_scratch.py` `imdb_bert_native.py` `imdb_bert_trainer.py` `imdb_distilbert_native.py` `imdb_distilbert_trainer.py` `imdb_roberta_trainer.py` | HuggingFace 预训练模型微调，直接读 tsv，不依赖 pickle |

## Kaggle 运行顺序

1. **手机验证**（开 GPU 的唯一门槛）：Settings → Phone Verification
2. 比赛 `word2vec-nlp-tutorial` → New Notebook → **Add Data 搜 `glove840b300dtxt`**（作者 takuok，免下载）→ Accelerator 选 GPU T4 x2
3. 解压比赛数据 + 转换 GloVe（gensim4 已删除官方转换脚本，手动加表头）：

```python
!mkdir -p corpus/imdb pickle result
!cd corpus/imdb && unzip -o -q /kaggle/input/competitions/word2vec-nlp-tutorial/labeledTrainData.tsv.zip
!cd corpus/imdb && unzip -o -q /kaggle/input/competitions/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip
!cd corpus/imdb && unzip -o -q /kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip

src = "/kaggle/input/glove840b300dtxt/glove.840B.300d.txt"
dst = "/kaggle/working/glove.840B.300d.gensim.txt"
with open(src, encoding="utf-8") as f:
    n = sum(1 for _ in f)                     # 应为 2196017
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
    g.write(f"{n} 300\n")
    for line in f: g.write(line)
```

4. `imdb_process.py` 粘进 cell，GloVe 路径改为 `/kaggle/working/glove.840B.300d.gensim.txt`，跑出生成 pickle
5. 按序跑模型：`imdb_cnn.py` → `imdb_lstm.py` → `imdb_gru.py` → `imdb_attention_lstm.py` → `imdb_transformer.py` → 其余
6. BERT 系（Kaggle 可直连 HuggingFace，无需镜像）：先 `imdb_distilbert_trainer.py`

## 详细指南

地雷清单见 [FIXES.md](FIXES.md)；完整逐步教程见配套网页指南（IMDB 深度学习进阶指南）。

## 学习关注点

跑每个模型时记录 val acc / train acc / 时长，重点观察两个问题：**过拟合**（train-val 差距拉大的拐点）与**梯度消失**（为什么 LSTM 治 RNN、Transformer 又取代 RNN）。配套视频：zh-v2.d2l.ai 第 8/9/11/15 章。
