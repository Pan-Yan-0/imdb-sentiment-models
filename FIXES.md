# 必改地雷清单（Kaggle Notebook 环境）

1. **手机验证**：Kaggle GPU（T4×2/P100，每周 30h）需验证过的账号；备选 Colab 免费 T4 免验证
2. **GloVe 免下载**：Add Data 搜 `glove840b300dtxt`（作者 takuok）直接挂载；gensim 4 已删除 glove2word2vec → 手动加表头转换（README 第 3 步代码）
3. **GloVe 路径**：`imdb_process.py` 第 99 行 `os.path.join("g:\\", 'lib', ...)` → `"/kaggle/working/glove.840B.300d.gensim.txt"`
4. **数据路径**：GloVe 系脚本读 `./corpus/imdb/*.tsv`（解压版）→ unzip 到该目录；BERT 系可直接读 `/kaggle/input/competitions/.../*.tsv.zip`
5. **目录**：`mkdir -p pickle result` 别跳过
6. **pickle 复用**：process 跑完 Save Version，pickle 作为输出保留，后续模型 notebook 挂载复用
7. **gensim 4 API**：`model.vocab`/`model.syn0` → `model.wv.key_to_index`/`model.wv.vectors`（process 里 `index_to_key` 已是新写法 ✅）
8. **GPU**：脚本写死 `torch.device('cuda:0')`；调试时 Accelerator 关回 None 省额度；OOM 三连：batch 减半 → max_len 256 → 换 DistilBERT
9. `torch.autograd.Variable` 已废弃（warning 可忽略）
