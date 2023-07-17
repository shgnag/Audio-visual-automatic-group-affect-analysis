# Audio-visual Automatic Group Affect Analysis
This repository conatins the PyTorch implementation of <a href="https://ieeexplore.ieee.org/document/9511820">Audio-visual automatic group affect analysis</a> method.
## Dataset
For Video Level Group AFfect (VGAF) dataset contact - abhinav.dhall@monash.edu and emotiw2014@gmail.com 

## Training
```
python VGAFNet_fusion.py
```
This file need to change the path of the pre-processed features as an input.
For the holistic channel, frames are sampled from the original video. For the face-level channel, vggface features are extracted. Please refer the paper for more details on data pre-processing.

## Citation
If you find the code useful for your research, please consider citing our work:
```
@article{sharma2021audio,
  title={Audio-visual automatic group affect analysis},
  author={Sharma, Garima and Dhall, Abhinav and Cai, Jianfei},
  journal={IEEE Transactions on Affective Computing},
  year={2021},
  publisher={IEEE}
}
```
