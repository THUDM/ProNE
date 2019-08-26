# ProNE

### [Paper](https://www.ijcai.org/proceedings/2019/594)

ProNE: Fast and Scalable Network Representation Learning

Jie Zhang, [Yuxiao Dong](https://ericdongyx.github.io/), Yan Wang, [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/) and Ming Ding

Accepted to IJCAI 2019 Research Track!

## Prerequisites

- Linux or macOS
- Python 2 or 3
- scipy
- sklearn


## Installation

Clone this repo.

```bash
git clone https://github.com/lykeven/ProNE
cd ProNE
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

## Dataset

These datasets are public datasets.

- PPI contains 3,890 nodes and 76,584 edges.
- blogcatalog contains 10,312 nodes and 333,983 edges.
- youtube contains 1,138,499 nodes and 2,990,443 edges.

## Training

### Training on the existing datasets

Create emb directory to save output embedding file
```bash
mkdir emb
```
You can use `python proNE.py -graph example_graph` to train ProNE model on the example data.

If you want to train on the PPI dataset, you can run 

```bash
python proNE.py -graph data/PPI.ungraph -emb1 emb/PPI_sparse.emb -emb2 emb/PPI_spectral.emb
 -dimension 128 -step 10 -theta 0.5 -mu 0.2
```
Where PPI_sparse.emb and PPI_spectral.emb are output embedding files and dimension, step, theta and mu are our model parameters.


### Training on your own datasets

If you want to train ProNE on your own dataset, you should prepare the following files:
- edgelist.txt: Each line represents an edge, which contains two tokens `<node1> <node2>` where each token is a number starting from 0.

### Training on c++ version ProNE
ProNE is mainly single-thread(except for the svd on small matrices). We also provide a c++ multi-thread program ProNE.cpp for large-scale network based on
 [Eigen](http://eigen.tuxfamily.org), [redsvd](https://code.google.com/p/redsvd/) and [boost](https://www.boost.org/). [Openmp](https://www.openmp.org/) and [MKL](https://software.intel.com/en-us/mkl) are used to speed up. Besides, [gflags](https://github.com/gflags/gflags) is required to parse command parameter.
This version is about 3 times faster under all optimization than the reported result in paper on youtube and the performance is still optimizing. 

Compile it via
```bash
g++ ProNE.cpp -fopenmp -l redsvd -l gflags -l pthread -o3 -o ProNE.out
```

If you want to train on the PPI dataset, you can run
```bash
./ProNE.out -filename data/PPI.ungraph -emb1 emb/PPI.emb1 -emb2 emb/PPI.emb2
 -num_node 3890 -num_step 10 -num_thread 20 -num_rank 128 -theta 0.5 -mu 0.2
```


If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.


## Citing

If you find *ProNE* is useful for your research, please consider citing our paper:

```
@inproceedings{ijcai2019-594,
  title     = {ProNE: Fast and Scalable Network Representation Learning},
  author    = {Zhang, Jie and Dong, Yuxiao and Wang, Yan and Tang, Jie and Ding, Ming},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {4278--4284},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/594},
  url       = {https://doi.org/10.24963/ijcai.2019/594},
}
```
