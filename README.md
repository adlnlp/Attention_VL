# Understanding Attention for Vision-and-Language Tasks
This repository contains code for the paper [Understanding Attention for Vision-and-Language Tasks](https://arxiv.org/abs/2208.08104) published in **COLING 2022**.

### <div align="center">Feiqi Cao, Soyeon Caren Han, Siqu Long, Changwei Xu, Josiah Poon. (2022, October).<br>[Understanding Attention for Vision-and-Language Tasks](https://arxiv.org/abs/2208.08104)<br><br>The 29th International Conference on Computational Linguistics <br> (COLING 2022).</div>

## Set Up
This paper analyzes the effect of different attention alignment calculation scores based on the following four Vision-and-Language (VL) tasks. We follow the instructions from their respective repositories to set up the environment and prepare the datasets.

- Text-to-Image Generation: [AttnGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) ([Github](https://github.com/taoxugit/AttnGAN))
- Text-and-Image Matching: [SCAN](https://arxiv.org/pdf/1803.08024.pdf) ([Github](https://github.com/kuanghuei/SCAN))
- Visual Question Answering: [MAC](https://arxiv.org/pdf/1803.03067.pdf) ([Github](https://github.com/stanfordnlp/mac-network))
- Text-based Visual Question Answering: [M4C](https://arxiv.org/pdf/1911.06258.pdf) (Please take note that we referred to the base code of [SAM4C Github](https://github.com/yashkant/sam-textvqa) and modified the config to include only classic Self-Attention Layers in the model, which becomes identical to the structure of M4C model)

## Run Experiments
The codes in our repository have the attention calculation part modified for each of the above base models. We provide the instructions for running our codes/experiments:

- Text-to-Image Generation: 
  - [experiment source code and configs](https://github.com/adlnlp/Attention_VL/tree/main/AttnGan/source_code)
  - [sample tutorial to run experiments on MSCOCO](https://github.com/adlnlp/Attention_VL/blob/main/AttnGan/attnGan_coco.ipynb)
  - [sample tutorial to run experiments on CUB](https://github.com/adlnlp/Attention_VL/blob/main/AttnGan/attnGAN_birds.ipynb)
- Text-and-Image Matching: 
  - [experiment source code and configs](https://github.com/adlnlp/Attention_VL/tree/main/SCAN/source%20code)
  - [sample tutorial to run experiments on MSCOCO](https://github.com/adlnlp/Attention_VL/blob/main/SCAN/run_MScoco.ipynb)
  - [sample tutorial to run experiments on Flickr30k](https://github.com/adlnlp/Attention_VL/blob/main/SCAN/run_F30k.ipynb)
- Visual Question Answering: 
  - [experiment source code and configs](https://github.com/adlnlp/Attention_VL/tree/main/MAC/source_code)
  - [sample tutorial to run experiments on CLEVR](https://github.com/adlnlp/Attention_VL/blob/main/MAC/mac.ipynb)
- Text-based Visual Question Answering: 
  - [experiment source code and configs](https://github.com/adlnlp/Attention_VL/tree/main/M4C)
  - sample commands to run experiments on Text-VQA:
  ```
  python train.py --config ./configs/m4c_tvqa_n4.yml --tag scaled_dot
  ```
  ```
  python train.py --config ./configs/m4c_tvqa_n4_dot.yml --tag dot
  ```
  ```
  python train.py --config ./configs/m4c_tvqa_n4_kwq.yml --tag general_kwq
  ```
  ```
  python train.py --config ./configs/m4c_tvqa_n4_biased_kwq.yml --tag biased_general_kwq
  ```
  ...

## Citation
```
@inproceedings{cao2022attentionvl,
  title     = {Understanding Attention for Vision-and-Language Tasks},
  author    = {Cao, Feiqi and Han, Soyeon Caren and Long, Siqu and Xu, Changwei, and Poon, Josiah},
  booktitle = {Proceedings of the 30th International Conference on Computational Linguistics},
  publisher = {International Committee on Computational Linguistics},
  month     = {oct},
  year      = {2022}
}
```

## Qualitative Examples

We visualised the prediction interpretability of the best and worst attention alignment calculation method for each task. Here are some examples. For more details please refer to our paper.

- Text-to-Image Generation: 
  <p align="center"><img src="figures/AttnGAN_analysis_1.jpg" width="600" /></p>
  <p align="center"><img src="figures/AttnGAN_analysis_2.jpg" width="600" /></p>
- Text-and-Image Matching: 
  <p align="center"><img src="figures/SCAN_analysis_1.jpg" width="600" /></p>
  <p align="center"><img src="figures/SCAN_analysis_2.jpg" width="600" /></p>
- Visual Question Answering: 
  <p align="center"><img src="figures/MAC_analysis_1.jpg" width="600" /></p>
  <p align="center"><img src="figures/MAC_analysis_2.jpg" width="600" /></p>
- Text-based Visual Question Answering: 
  <p align="center"><img src="figures/M4C_analysis_1.jpg" width="600" /></p>
  <p align="center"><img src="figures/M4C_analysis_2.jpg" width="600" /></p>
