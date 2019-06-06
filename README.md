# ProNE

### [Arxiv](https://arxiv.org/abs/1806.02623)

ProNE: Fast and Scalable Network Representation Learning

Jie Zhang, [Yuxiao Dong](https://ericdongyx.github.io/), Yan Wang, [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/) and Ming Ding

Accepted to IJCAI 2019 Research Track!

## Prerequisites

- Linux or macOS
- Python 2 or 3
- scipy
- sklearn

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/lykeven/ProNE
cd ProNE
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

These datasets are public datasets.

- PPI contains 3,890 nodes and 76,584 edges.
- blogcatalog contains 10,312 nodes and 333,983 edges.
- youTube contains 1,138,499 nodes and 2,990,443 edges.

### Training

#### Training on the existing datasets

You can use `python proNE.py --graph example_graph` to train ProNE model on the example data.

If you want to train on the PPI dataset, you can run 

```bash
python proNE.py -graph data/PPI.ungraph -emb1 PPI_sparse.emb -emb2 PPI_spectral.emb
 -dimension 128 -step 10 -theta 0.5 -mu 0.2
```
Where PPI_sparse.emb and PPI_spectral.emb are output embedding files and dimension, step, theta and mu are our model parameters.

#### Training on your own datasets

If you want to train ProNE on your own dataset, you should prepare the following files:
- edgelist.txt: Each line represents an edge, which contains two tokens `<node1> <node2>` where each token is a number starting from 0.

If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.

## Cite

Please cite our paper if you find this code useful for your research:

```
@article{zhang2018spectral,
  title={Spectral Network Embedding: A Fast and Scalable Method via Sparsity},
  author={Zhang, Jie and Wang, Yan and Tang, Jie},
  journal={arXiv preprint arXiv:1806.02623},
  year={2018}
}
```