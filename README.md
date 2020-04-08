# TALL.pytorch

Pytorch implementation of [_TALL: Temporal Activity Localization via Language Query_](https://arxiv.org/abs/1705.02101)

Pytorch==1.4.0

Torchvision==0.5.0

[Official Implementaion](https://github.com/jiyanggao/TALL) in Tensorflow

**Note: Still very slow and needs optimisations.**

### My Results

|                  | IoU=0.1     | IoU=0.3     | IoU=0.5     | IoU=0.7     |
| :--------------- | ----------: | ----------: | ----------: | ----------: | 
| R@1              |   23.90     |    18.85    |    14.10    |     7.44    |
| R@5              |   48.46     |    36.00    |    25.05    |     14.79   |
| R@10             |   62.55     |    44.64    |    30.36    |     18.12   |

### Visual Features on TACoS
Download the C3D features for [training set](https://drive.google.com/file/d/1zQp0aYGFCm8PqqHOh4UtXfy2U3pJMBeu/view?usp=sharing)  and [test set](https://drive.google.com/file/d/1zC-UrspRf42Qiu5prQw4fQrbgLQfJN-P/view?usp=sharing) of TACoS dataset.

### Sentence Embeddings on TACoS
Download the Skip-thought sentence embeddings and sample files from [here](https://drive.google.com/file/d/1HF-hNFPvLrHwI5O7YvYKZWTeTxC5Mg1K/view?usp=sharing) of TACoS Dataset, and put them under exp_data folder.

