# Understanding Attention for Vision-and-Language Tasks
This repository contains code for the paper [Understanding Attention for Vision-and-Language Tasks](https://arxiv.org/abs/2208.08104) published in **COLING 2022**.

__<p align="center">Feiqi Cao, Soyeon Caren Han, Siqu Long, Changwei Xu, Josiah Poon</p>__

## Set Up
This paper analyzes the effect of different attention alignment calculation scores based on the following four Vision-and-Language (VL) tasks. We follow the instructions from their respective repositories to set up the environment and prepare the datasets.

- Text-to-Image Generation: [AttnGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) ([Github](https://github.com/taoxugit/AttnGAN))
- Text-and-Image Matching: [SCAN](https://arxiv.org/pdf/1803.08024.pdf) ([Github](https://github.com/kuanghuei/SCAN))
- Visual Question Answering: [MAC](https://arxiv.org/pdf/1803.03067.pdf) ([Github](https://github.com/stanfordnlp/mac-network))
- Text-to-Image Generation: [M4C](https://arxiv.org/pdf/1911.06258.pdf) (Please take note that we referred to the base code of [SAM4C Github](https://github.com/yashkant/sam-textvqa) and modified the config to include only classic Self-Attention Layers in the model, which becomes identical to the structure of M4C model)

## Run Experiments
The codes in our repository have the attention calculation part modified for each of the above base models. We provide the instructions for running our codes/experiments:

- Text-to-Image Generation: 
- Text-and-Image Matching: 
- Visual Question Answering: 
- Text-to-Image Generation: 
