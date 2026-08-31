# IMDB 情感分类 · 深度学习模型集

文本分类技术演进线的后半段：从 TextCNN / RNN 系一路到 BERT 系。
配套前置仓库：[word2vec-nlp-tutorial](https://github.com/Pan-Yan-0/word2vec-nlp-tutorial)（词袋 / Word2Vec 阶段，分数 0.846 / 0.844）。

## 结构

| 组 | 文件 | 说明 |
|---|---|---|
| 预处理 | `imdb_process.py` | 清洗 + 建词表 + 加载 GloVe → 生成 `pickle/imdb_glove.pickle3`（先跑且只跑一次） |
| GloVe 系 | `imdb_cnn.py` `imdb_lstm.py` `imdb_gru.py` `imdb_attention_lstm.py` `imdb_cnnlstm.py` `imdb_capsule_lstm.py` `imdb_transformer.py` | 静态词向量 + PyTorch 手写网络，依赖 pickle |
| BERT 系 | `imdb_bert_scratch.py` `imdb_bert_native.py` `imdb_bert_trainer.py` `imdb_distilbert_native.py` `imdb_distilbert_trainer.py` `imdb_roberta_trainer.py` | HuggingFace 预训练模型微调，直接读 tsv，不依赖 pickle |

## 运行顺序

```bash
mkdir -p corpus/imdb pickle result
# 1. 数据：labeledTrainData.tsv / unlabeledTrainData.tsv / testData.tsv 解压到 ./corpus/imdb/
# 2. GloVe：下载 glove.840B.300d.zip 解压后用 tools/convert_glove.py 转换（gensim4 已删除官方转换脚本）
python tools/convert_glove.py glove.840B.300d.txt glove.840B.300d.gensim.txt
# 3. 改 imdb_process.py 里的 GloVe 路径（原写死 g:\lib\），然后：
python imdb_process.py
# 4. 跑模型（建议顺序）
python imdb_cnn.py && python imdb_lstm.py && python imdb_gru.py
python imdb_attention_lstm.py && python imdb_transformer.py
# 5. BERT 系（国内先 export HF_ENDPOINT=https://hf-mirror.com）
python imdb_distilbert_trainer.py
```

## 已知地雷（详见 FIXES.md）

- `imdb_process.py` 中 GloVe 路径写死为 `g:\lib\`，必须改
- 数据读的是解压后的 .tsv，不是 .zip
- gensim 4 删除 glove2word2vec 官方脚本 → 用 `tools/convert_glove.py`
- 全部脚本写死 `cuda:0`；无 GPU 环境需改 device；BERT 系 4GB 显存可能 OOM → 降 batch_size

## 学习关注点

跑每个模型时记录 val acc / train acc / 时长，重点观察两个问题：**过拟合**（train-val 差距拉大的拐点）与**梯度消失**（为什么 LSTM 治 RNN、Transformer 又取代 RNN）。配套视频：zh-v2.d2l.ai 第 8/9/11/15 章。
