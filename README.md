# DSWN

Implementation of our sRGB denoising method in CVPR 2020 NTIRE Workshop

DSWN is an augmentation method based on SGN in ICCV 2019. It introduces DWT / IDWT in network architecture. Besides, an ensemble method is utilized to improve final performance.

Install `pytorch_wavelets` lib before use:
```bash
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```

If it is succussfully installed, you will obtain a feedback:
```bash
Successfully built pytorch-wavelets
Installing collected packages: pytorch-wavelets
Successfully installed pytorch-wavelets-1.2.2
```

Then run DSWN:
```bash
cd ..
cd DSWN
python train.py
(please make sure that using the DIV2K dataset as training set)
```

If you think this page is helpful for your research, please consider cite:
```bash
@inproceedings{liu2020densely,
  title={Densely Self-Guided Wavelet Network for Image Denoising},
  author={Liu, Wei and Yan, Qiong and Zhao, Yuzhi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={432--433},
  year={2020}
}
```
