# UniTRec
This repository releases the code of paper [**UniTRec: A Unified Text-to-Text Transformer and Joint Contrastive Learning Framework for Text-based Recommendation** (ACL-2023 Short Paper)](https://aclanthology.org/2023.acl-short.100.pdf).
<br/><br/>


## Dataset Preparation
Our code will download and pre-tokenize the datasets automatically. Also refer to [setup.sh](https://github.com/Veason-silverbullet/UniTRec/blob/master/setup.sh).
<pre><code>cd textRec_datasets
python newsrec_tokenize.py
python quoterec_tokenize.py
python engagerec_tokenize.py</code></pre>


## UniTRec Training
Suppose that two GPUs are available for training.
<pre><code>CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 newsrec.py</code></pre>
<pre><code>CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 quoterec.py</code></pre>
<pre><code>CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 engagerec.py</code></pre>


## Note
The transformer codebase is adapted from [**Huggingface Transoformers**](https://github.com/huggingface/transformers). The **UniTRec Model** is written at [transformers/models/UniTRec/modeling_unitrec.py](https://github.com/Veason-silverbullet/UniTRec/blob/master/transformers/models/UniTRec/modeling_unitrec.py).
<br/><br/>


## TODO
1. The codes are now using two GPUs for training and one for inference. Acceleration can be achieved by distributed inference.

2. I plan to release baseline codes, but they are on my rented cloud machines and need time to arrange. As my physical and mental states are quite bad, I try my best to use spare time to pull down, arrange, and release the baseline codes ASAP.


## Citation
```
@inproceedings{mao-etal-2023-unitrec,
    title = "UniTRec: A Unified Text-to-Text Transformer and Joint Contrastive Learning Framework for Text-based Recommendation",
    author = "Mao, Zhiming  and
              Wang, Huimin  and
              Du, Yiming  and
              Wong, Kam-Fai",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.100",
    pages = "1160--1170"
}
```
