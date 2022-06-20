---
tasks:
- acoustic-noise-suppression
model_type:
- complex-nn
domain:
- speech
frameworks:
- pytorch
model-backbone:
- frcrn
license: apache-2.0
tags:
- Alibaba
- Mind DNS
---


# 语音智能降噪介绍

音频通话场景和各种噪声环境下语音音频录音的单通道语音智能降噪模型算法

## 模型描述

模型输入和输出均为16kHz采样率单通道语音时域波形信号，输入信号可有单通道麦克风直接进行录制，输出为噪声抑制后的语音音频信号[1]。模型采用Deep Complex CRN结构，模型输入信号通过STFT变换转换成复数频谱特征作为输入，并采用Complex FSMN在频域上进行关联性处理和在时序特征上进行长序处理，预测中间输出目标Complex ideal ratio mask, 然后使用预测的mask和输入频谱相乘后得到增强后的频谱，最后通过STFT逆变换得到增强后语音波形信号。模型的训练数据采用了DNS-Challenge开源数据集[2]。

### 模型局限性以及可能的偏差

模型在存在多说话人干扰声的场景噪声抑制性能有不同程度的下降。

### 相关论文以及引用信息

[1] Shengkui Zhao, Bin Ma, Karn N. Watcharasupat, and Woon-Seng Gan. "FRCRN: Boosting Feature Representation Using Frequency Recurrence for Monaural Speech Enhancement." In ICASSP 2022. IEEE. May 2022.

[2] Harishchandra Dubey, Vishak Gopal, Ross Cutler, Ashkan Aazami, Sergiy Matusevych, Sebastian Braun, Sefik Emre Eskimez, Manthan Thakker, Takuya Yoshioka, Hannes Gamper, and Robert Aichner. "ICASSP 2022 Deep Noise Suppression Challenge." In ICASSP 2022, IEEE. May 2022.