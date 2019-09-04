# Batch DropBlock Network for Person Re-identification and Beyond
Official source code of paper https://arxiv.org/abs/1811.07130

## Update on 2019.3.15
Update CUHK03 results. 

## Update on 2019.1.29
Traning scripts are released. The best Markt1501 result is 95.3%! Please look at the training section of README.md.

## Update on 2019.1.23
In-Shop Clothes Retrieval dataset and pretrained model are released!. The rank-1 result is 89.5 which is a litter bit higher than paper reported.

## This paper is accepted by ICCV 2019. Please cite if you use this code in your research. 

```
@article{dai2018batch,
  title={Batch DropBlock Network for Person Re-identification and Beyond},
  author={Dai, Zuozhuo and Chen, Mingqiang and Gu, Xiaodong and Zhu, Siyu and Tan, Ping},
  journal={arXiv preprint arXiv:1811.07130},
  year={2018}
}
```

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

    For In-Shop Clothes dataset,
    1. Downlaod clothes dataset from http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/clothes.tar
    2. Extract dataset and put it to `data/` folder.

## Results

Dataset | CUHK03-Label | CUHK03-Detect | DukeMTMC re-ID  | Market1501 | In-Shop Clothes|
--------|--------------|---------------|-----------------|------------|----------------|
Rank-1  | 79.4         | 76.4          | 88.9            | 95.3       |89.5            |
mAP     | 76.7         | 73.5          | 75.9            | 86.2       |72.3            |
model   | [aliyun](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/cuhk-label-794.pth.tar)| [aliyun](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/cuhk-detect-764.pth.tar)] | [aliyun](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/duke_887.pth.tar) | [aliyun](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/market_953.pth.tar)|[aliyun](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/clothes_895.pth.tar)

You can download the pre-trained models from the above table and evaluate on person re-ID datasets.
For example, to evaluate CUHK03-Label dataset, you can download the model to './pytorch-ckpt/cuhk_label_bfe' directory and run the following commands.

### Evaluate Market1501
```bash
python3 main_reid.py train --save_dir='./pytorch-ckpt/market_bfe' --model_name=bfe --train_batch=32 --test_batch=32 --dataset=market1501 --pretrained_model='./pytorch-ckpt/market_bfe/944.pth.tar' --evaluate
```
### Evaluate CUHK03-Label
```bash
python3 main_reid.py train --save_dir='./pytorch-ckpt/cuhk_label_bfe' --model_name=bfe --train_batch=32 --test_batch=32 --dataset=cuhk-label  --pretrained_model='./pytorch-ckpt/cuhk_label_bfe/750.pth.tar' --evaluate
```
### Evaluate In-Shop clothes
```bash
python main_reid.py train --save_dir='./pytorch-ckpt/clothes_bfe' --model_name=bfe --pretrained_model='./pytorch-ckpt/clothes_bfe/clothes_895.pth.tar' --test_batch=32 --dataset=clothes --evaluate
```

## Training

### Traning Market1501
```bash
python main_reid.py train --save_dir='./pytorch-ckpt/market-bfe' --max_epoch=400 --eval_step=30 --dataset=market1501 --test_batch=128 --train_batch=128 --optim=adam --adjust_lr
```
This traning command is tested on 4 GTX1080 gpus. Here is [training log](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/bfe_models/market_953.txt). You shoud get a result around 95%.
