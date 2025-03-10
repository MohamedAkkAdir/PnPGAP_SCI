# Plug-and-play Algorithms for Snapshot Compressive Imaging (PnP-SCI) [pre-release]
Yang Liu, MIT CSAIL, yliu@csail.mit.edu, updated July 2, 2020 

This repository contains the Python (PyTorch) code for the paper **Plug-and-play Algorithms for Large-scale Snapshot Compressive Imaging** in ***IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*** 2020 (**Oral**) by [Xin Yuan](https://www.bell-labs.com/usr/x.yuan), [Yang Liu](https://liuyang12.github.io/), [Jinli Suo](https://sites.google.com/site/suojinli/), and [Qionghai Dai](http://media.au.tsinghua.edu.cn/).
[[pdf]](https://arxiv.org/pdf/2003.13654 "arXiv preprint")   [[MATLAB code]](https://github.com/liuyang12/PnP-SCI "github repository for MATLAB code")   [[arXiv]](https://arxiv.org/abs/2003.13654 "arXiv preprint"). 

The initial Python code for GAP-TV was from [Dr. Xin Yuan](https://www.bell-labs.com/usr/x.yuan) on Aug 7, 2018.


## How to run this code
This code is tested on Google colab with CUDA-enabled GPU enabled. It is supposed to work on other platforms (Linux or Windows) with CUDA-enabled GPU(s). 

We use [conda](https://www.anaconda.com/distribution/) to manage the virtual environment and Python packages.

In google colab:

Note: Don't forget to enable GPU acceleration in Google Colab.

0. Open the notebook `pnp_sci_demo.ipynb` in Google Colab via the link : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohamedAkkAdir/PnPGAP_SCI/blob/main/pnp_sci_demo.ipynb)
1. follow the instructions in the notebook.
2. [Optional] Explore more with the main algorithm code `pnp_sci_algo.py`.


## Acknowledgements
We adopt several image and video denoisers as video priors (under the `./packages/` directory). The `ffdnet` package is from the (unofficial) PyTorch implementation of [FFDNet, TIP'18](https://doi.org/10.1109/TIP.2018.2839891) at https://doi.org/10.5201/ipol.2019.231 (An official PyTorch implementation of FFDNet was out just recently at https://github.com/cszn/KAIR. We believe they should work similarly with respect to the reconstruction process). The `fastdvdnet` package is from the PyTorch implementation of [FastDVDnet](https://arxiv.org/abs/1907.01361) at https://github.com/m-tassano/fastdvdnet.

## Citation
```
@inproceedings{Yuan2020PnPSCI,
  title     = {Plug-and-Play Algorithms for Large-scale Snapshot Compressive Imaging},
  author    = {Yuan, Xin and Liu, Yang and Suo, Jinli and Dai, Qionghai},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1447--1457},
  year      = {2020}
}
```