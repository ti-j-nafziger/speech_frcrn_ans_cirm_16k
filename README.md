---
tasks:
- acoustic-noise-suppression

widgets:
- task: acoustic-noise-suppression
  inputs:
  - type: audio
    name: input
    title: 带噪音的原始音频
    validator:
      max_size:10M
  examples:
  - name: 1
    title: 示例1
    inputs:
    - name: input
      data: git://examples/speech_with_noise.wav
  inferencespec:
    cpu: 1
    memory: 1000
    gpu: 0
    gpu_memory: 1000
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

模型输入和输出均为16kHz采样率单通道语音时域波形信号，输入信号可由单通道麦克风直接进行录制，输出为噪声抑制后的语音音频信号[1]。模型采用Deep Complex CRN结构，模型输入信号通过STFT变换转换成复数频谱特征作为输入，并采用Complex FSMN在频域上进行关联性处理和在时序特征上进行长序处理，预测中间输出目标Complex ideal ratio mask, 然后使用预测的mask和输入频谱相乘后得到增强后的频谱，最后通过STFT逆变换得到增强后语音波形信号。模型的训练数据采用了DNS-Challenge开源数据集[2]。

## 期望模型使用方式以及适用范围


### 如何使用

在安装ModelScope完成之后即可使用```speech_frcrn_ans_cirm_16k```进行推理。模型输入和输出均为16kHz采样率单通道语音时域波形信号，输入信号可由单通道麦克风直接进行录制，输出为噪声抑制后的语音音频信号。为了方便使用在pipeline在模型处理前后增加了wav文件处理逻辑，可以直接读取一个wav文件，并把输出结果保存在指定的wav文件中。

#### 环境准备：

* 本模型已经在1.8~1.11下测试通过，由于PyTorch v1.12.0的[BUG](https://github.com/pytorch/pytorch/issues/80837)，目前无法在v1.12.0上运行，如果您已经安装了此版本请执行以下命令回退到v1.11

```
conda install pytorch==1.11 torchaudio torchvision -c pytorch
```

* 本模型的pipeline中使用了三方库SoundFile进行wav文件处理，**在Linux系统上用户需要手动安装SoundFile的底层依赖库libsndfile**，在Windows和MacOS上会自动安装不需要用户操作。详细信息可参考[SoundFile官网](https://github.com/bastibe/python-soundfile#installation)。以Ubuntu系统为例，用户需要执行如下命令:

```shell
sudo apt-get update
sudo apt-get install libsndfile1
```

#### 代码范例

```python
ans = pipeline(
   Tasks.speech_signal_process,
   model='damo/speech_frcrn_ans_cirm_16k',
   pipeline_name=r'speech_frcrn_ans_cirm_16k')
ans('speech_with_noise.wav', output_path='output.wav')
```

### 模型局限性以及可能的偏差

模型在存在多说话人干扰声的场景噪声抑制性能有不同程度的下降。

## 数据评估及结果

模型效果请参考下面相关论文。

### 相关论文以及引用信息

[1] Shengkui Zhao, Bin Ma, Karn N. Watcharasupat, and Woon-Seng Gan. "FRCRN: Boosting Feature Representation Using Frequency Recurrence for Monaural Speech Enhancement." In ICASSP 2022. IEEE. May 2022.

[2] Harishchandra Dubey, Vishak Gopal, Ross Cutler, Ashkan Aazami, Sergiy Matusevych, Sebastian Braun, Sefik Emre Eskimez, Manthan Thakker, Takuya Yoshioka, Hannes Gamper, and Robert Aichner. "ICASSP 2022 Deep Noise Suppression Challenge." In ICASSP 2022, IEEE. May 2022.
