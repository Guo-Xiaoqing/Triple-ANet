# Triple ANet: Adaptive Abnormal-aware Attention Network for WCE Image Classification

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/), [Yixuan Yuan](http://www.ee.cityu.edu.hk/~yxyuan/people/people.htm).

## Summary:
### Intoduction:
This repository is for our MICCAI2019 paper ["Triple ANet: Adaptive Abnormal-aware Attention Network for WCE Image Classification"]()
### Framework:
![](https://github.com/Guo-Xiaoqing/Triple-ANet/raw/master/image/framework.png)

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
Use "make_txt.py" to split training data and testing data. The generated txt files are showed in folder "txt".
"make_tfrecords.py" is used to make tfrecord formate data.

### Train the model: 
```
python3 Triple_ANet_train.py -tfdata_path ./tfrecord/
```

### Test the model: 
```
python3 Triple_ANet_test.py -tfdata_path ./tfrecord/
```
## Results:
![](https://github.com/Guo-Xiaoqing/Triple-ANet/raw/master/image/result.png)

## Citation:
To be updated

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
