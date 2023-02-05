# Scale up with Order: Finding Good Data Permutations for Distributed Training
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache-2.svg)](https://opensource.org/licenses/Apache-2.0)

D-GraB is a distributed gradient balancing framework that aims to find distributed data permutation with provably better convergence guarantees than Distributed Random Reshuffling (D-RR). Our paper can be found [here](https://arxiv.org/abs/2302.00845).


# Install
## Requirements
Python >= 3.7
PyTorch >= 1.10.0
CUDA >= 10.1 on linux

## Experiments

### LR with BERT embeddings on GLUE tasks

### LeNet on CIFAR10

### LSTM on Wiki2

### Random vectors simulation on herding bound


# Authors
 - Wentao Guo, wg247@cornell.edu
 - Khiem Pham, dkp45@cornell.edu
 - [Yucheng Lu](https://www.cs.cornell.edu/~yucheng/), yl2967@cornell.edu
 - Tiancheng Yuan, ty373@cornell.edu 
 - Charlie F. Ruan, cfr54@cornell.edu
 - [Christopher De Sa](https://www.cs.cornell.edu/~cdesa/), cdesa@cs.cornell.edu


# License
D-GraB uses Apache-2 license in the [LICENSE](https://github.com/GarlGuo/D-GraB/tree/release/LICENSE) file.


# Cite us

If you find D-GraB helpful in your research, please consider citing us:

```
@misc{https://doi.org/10.48550/arxiv.2302.00845,
    doi = {10.48550/ARXIV.2302.00845},
    url = {https://arxiv.org/abs/2302.00845},
    author = {Guo, Wentao and Pham, Khiem and Lu, Yucheng and Yuan, Tiancheng and Ruan, Charlie F. and De Sa, Christopher},
    keywords = {Machine Learning (cs.LG), Distributed, Parallel, and Cluster Computing (cs.DC), Optimization and Control (math.OC), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Mathematics, FOS: Mathematics},
    title = {Scale up with Order: Finding Good Data Permutations for Distributed Training},
    publisher = {arXiv},
    year = {2023},
    copyright = {arXiv.org perpetual, non-exclusive license}
}
```