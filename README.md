# Batch Feature Erasing for Person Re-identification and Beyond
Official source code of paper https://arxiv.org/abs/1811.07130

## Setup running environment
This project requires python3, cython, torch, torchvision, scikit-learn, tensorboardX, fire.
The baseline source code is borrowed from https://github.com/L1aoXingyu/reid_baseline.
## Prepare dataset
    
    Create a directory to store reid datasets under this repo via
    ```bash
    cd reid
    mkdir data
    ```
    For market1501 dataset, 
    1. Download Market1501 dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html
    2. Extract dataset and rename to `market1501`. The data structure would like:
    ```
    market1501/
        bounding_box_test/
        bounding_box_train/
        query/
    ```
    For CUHK03 dataset,
    1. Download CUHK03-NP dataset from https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP 
    2. Extract dataset and rename folers inside it to cuhk-detect and cuhk-label.
    For DukeMTMC-reID dataset,
    Dowload from https://github.com/layumi/DukeMTMC-reID_evaluation

## Results

Dataset | CUHK03-Label | CUHK03-Detect | DukeMTMC re-ID  | Market1501 |
--------|--------------|---------------|-----------------|------------|
Rank-1  | 75.0         | 72.1          | 88.7            | 94.4       |
mAP     | 70.9         | 67.9          | 75.8            | 85.0       |
model   |[aliyun](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/cuhk_label_750.pth.tar) | [aliyun](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/cuhk_detect_720.pth.tar) | [aliyun](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/duke_887.pth.tar) | [aliyun](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/market_944.pth.tar)

You can download the pre-trained models from the above table and evaluate on person re-ID datasets.
For example, to evaluate CUHK03-Label dataset, you can download the model to './pytorch-ckpt/cuhk_label_bfe' directory and run the following command:

```bash
python3 main_reid.py train --save_dir='./pytorch-ckpt/cuhk_label_bfe' --model_name=bfe --train_batch=32 --test_batch=32 --dataset=cuhk-label  --pretrained_model='./pytorch-ckpt/cuhk_label_bfe/750.pth.tar' --evaluate
```