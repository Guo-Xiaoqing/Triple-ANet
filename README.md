# Triple ANet: Adaptive Abnormal-aware Attention Network for WCE Image Classification

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/), [Yixuan Yuan](http://www.ee.cityu.edu.hk/~yxyuan/people/people.htm).

## Summary:
### Intoduction:
This repository is for our MICCAI2019 paper ["Triple ANet: Adaptive Abnormal-aware Attention Network for WCE Image Classification"](https://link.springer.com/content/pdf/10.1007%2F978-3-030-32239-7_33.pdf)
### Framework:
![](https://github.com/Guo-Xiaoqing/Triple-ANet/raw/master/framework.png)

## Usage:
### Requirement:
Tensorflow 1.4
Python 3.5

### Preprocessing:
Clone the repository:
```
git clone https://github.com/Guo-Xiaoqing/Triple-ANet.git
cd Triple-ANet
```
Use "make_txt.py" to split training data and testing data. The generated txt files are showed in folder "./txt/".

"make_tfrecords.py" is used to make tfrecord format data, which could be stored in folder "./tfrecord/".

#### Update at 2020.05.06
To avoid additional data augmentation before training, online data augmentations, including random flips and rotations, are added in script "utilsForTF.py".

### Train the model: 
```
python3 Triple_ANet_train.py --tfdata_path ./tfrecord/
```

### Test the model: 
```
python3 Triple_ANet_test.py --tfdata_path ./tfrecord/
```
## Results:
![](https://github.com/Guo-Xiaoqing/Triple-ANet/raw/master/result.png)
From top to bottom, they are respectively inflammatory, vascular lesion and polyp samples. (a) Original image. (b)(c) show attention maps of the 1st and 2nd AAM. (d)(e) show offset fields in the 1st branch and 2nd branch of the 1st AMM while (f)(g) show offset fields in the 1st branch and 2nd branch of the 2st AMM. (h) Ground truth of mask.

## Citation:
If you found Triple ANet helpful for your research, please cite our paper:
```
@inproceedings{guo2019triple,
  title={Triple ANet: Adaptive Abnormal-aware Attention Network for WCE Image Classification},
  author={Guo, Xiaoqing and Yuan, Yixuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={293--301},
  year={2019},
  organization={Springer}
}
```

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
