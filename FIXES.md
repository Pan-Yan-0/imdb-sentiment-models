# 必改地雷清单

1. **GloVe 路径**：`imdb_process.py` 第 99 行 `os.path.join("g:\\", 'lib', 'glove.840B.300d.gensim.txt')` → 改成实际路径，如 `"./glove.840B.300d.gensim.txt"`
2. **数据路径**：脚本读 `./corpus/imdb/*.tsv`（解压后的 tsv），不是 Kaggle 的 .zip
3. **目录**：`mkdir -p pickle result`（process 写 pickle/，distilbert_native 写 result/）
4. **gensim 4**：官方 `glove2word2vec` 已删除 → 用 `tools/convert_glove.py`；旧 API `model.vocab`/`model.syn0` 对应新写法 `model.wv.key_to_index`/`model.wv.vectors`
5. **HuggingFace 国内下载**：`export HF_ENDPOINT=https://hf-mirror.com`
6. **GPU**：脚本写死 `torch.device('cuda:0')`；无卡调试改 `'cuda:0' if torch.cuda.is_available() else 'cpu'`
7. **OOM 三连**：batch_size 减半 → max_len 512→256 → 换 DistilBERT
8. `torch.autograd.Variable` 已废弃（warning 可忽略，或替换为普通 tensor）
