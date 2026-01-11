# TAD

[![arXiv](https://img.shields.io/badge/arXiv-2303.05092-b31b1b.svg)](https://arxiv.org/abs/2303.05092)

- This is the official implementation for [Task Aware Dreamer for Task Generalization in Reinforcement Learning]([https://www.ijcai.org/proceedings/2022/0510.pdf](https://arxiv.org/abs/2303.05092)).

- The training code is based on [dreamer-pytorch](https://github.com/yusukeurakami/dreamer-pytorch).

## Usage

```sh
conda create -n TAD python=3.8
conda activate TAD

pip install --upgrade pip
pip install wheel==0.38.4 setuptools==66.0.0

pip install -r requirements.txt
```

## Citation

If you find this work helpful, please cite our paper.

```
@article{ying2023task,
  title={Task aware dreamer for task generalization in reinforcement learning},
  author={Ying, Chengyang and Hao, Zhongkai and Zhou, Xinning and Su, Hang and Liu, Songming and Yan, Dong and Zhu, Jun},
  journal={arXiv preprint arXiv:2303.05092},
  year={2023}
}
```
