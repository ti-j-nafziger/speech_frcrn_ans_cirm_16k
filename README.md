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
      max_size: 10M
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
- audio
frameworks:
- pytorch
model-backbone:
- frcrn
license: Apache License 2.0
tags:
- Alibaba
- Mind DNS
datasets:
  train:
  - modelscope/ICASSP_2021_DNS_Challenge
  evaluation:
  - modelscope/ICASSP_2021_DNS_Challenge

---


# FRCRN语音降噪模型介绍

本模型提供音频通话场景和各种噪声环境下语音音频录音的单通道语音智能降噪模型算法。

## 模型描述

模型输入和输出均为16kHz采样率单通道语音时域波形信号，输入信号可由单通道麦克风直接进行录制，输出为噪声抑制后的语音音频信号[1]。模型采用Deep Complex CRN结构，模型输入信号通过STFT变换转换成复数频谱特征作为输入，并采用Complex FSMN在频域上进行关联性处理和在时序特征上进行长序处理，预测中间输出目标Complex ideal ratio mask, 然后使用预测的mask和输入频谱相乘后得到增强后的频谱，最后通过STFT逆变换得到增强后语音波形信号。

![model.png](description/model.png)

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
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')
result = ans(
    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise.wav',
    output_path='output.wav')
```

### 模型局限性以及可能的偏差

模型在存在多说话人干扰声的场景噪声抑制性能有不同程度的下降。

## 训练数据介绍

模型的训练数据来自DNS-Challenge开源数据集，是Microsoft团队为ICASSP相关挑战赛提供的，[官方网址](https://github.com/microsoft/DNS-Challenge)[2]。我们这个模型是用来处理16k音频，因此只使用了其中的fullband数据，并做了少量调整。为便于大家使用，我们把DNS Challenge 2020的数据集迁移在modelscope的[DatasetHub](https://modelscope.cn/datasets/modelscope/ICASSP_2021_DNS_Challenge/summary)上，用户可参照数据集说明文档下载使用。

## 模型训练流程

需要先按照数据集说明在本地生成训练数据，然后**更新以下代码中数据集路径（/your_local_path/ICASSP_2021_DNS_Challenge）**，才能正常训练。

```python
import os

from datasets import load_dataset

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

tmp_dir = f'./ckpt'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

hf_ds = load_dataset(
    '/your_local_path/ICASSP_2021_DNS_Challenge',
    'train',
    split='train')
mapped_ds = hf_ds.map(
    to_segment,
    remove_columns=['duration'],
    num_proc=8,
    batched=True,
    batch_size=36)
mapped_ds = mapped_ds.train_test_split(test_size=3000)
mapped_ds = mapped_ds.shuffle()
dataset = MsDataset.from_hf_dataset(mapped_ds)

kwargs = dict(
    model='damo/speech_frcrn_ans_cirm_16k',
    model_revision='beta',
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    work_dir=tmp_dir)
trainer = build_trainer(
    Trainers.speech_frcrn_ans_cirm_16k, default_args=kwargs)
trainer.train()
```

## 数据评估及结果

与其他SOTA模型在DNS Challenge 2020官方测试集上对比效果如下：

![matrix.png](description/matrix.png)

指标说明：

* PESQ (Perceptual Evaluation Of Speech Quality) 语音质量感知评估，是一种客观的、全参考的语音质量评估方法，得分范围在-0.5--4.5之间，得分越高表示语音质量越好。
* STOI (Short-Time Objective Intelligibility) 短时客观可懂度，反映人类的听觉感知系统对语音可懂度的客观评价，STOI 值介于0~1 之间，值越大代表语音可懂度越高，越清晰。
* SI-SNR (Scale Invariant Signal-to-Noise Ratio) 尺度不变的信噪比，是在普通信噪比基础上通过正则化消减信号变化导致的影响，是针对宽带噪声失真的语音增强算法的常规衡量方法。

DNS Challenge的结果列表在[这里](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2022/results/)。

### 模型评估代码
可通过如下代码对模型进行评估验证，我们在modelscope的[DatasetHub](https://modelscope.cn/datasets/modelscope/ICASSP_2021_DNS_Challenge/summary)上存储了DNS Challenge 2020的验证集，方便用户下载调用。

```python
import os
import tempfile

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

hf_ds = MsDataset.load(
    'ICASSP_2021_DNS_Challenge', split='test').to_hf_dataset()
mapped_ds = hf_ds.map(
    to_segment,
    remove_columns=['duration'],
    # num_proc=5, # Comment this line to avoid error in Jupyter notebook
    batched=True,
    batch_size=36)
dataset = MsDataset.from_hf_dataset(mapped_ds)
kwargs = dict(
    model='damo/speech_frcrn_ans_cirm_16k',
    model_revision='beta',
    train_dataset=None,
    eval_dataset=dataset,
    val_iters_per_epoch=125,
    work_dir=tmp_dir)

trainer = build_trainer(
    Trainers.speech_frcrn_ans_cirm_16k, default_args=kwargs)

eval_res = trainer.evaluate()
print(eval_res['avg_sisnr'])

```

更多详情请参考下面相关论文。

### 相关论文以及引用信息

[1]

```BibTeX
@INPROCEEDINGS{9747578,
  author={Zhao, Shengkui and Ma, Bin and Watcharasupat, Karn N. and Gan, Woon-Seng},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={FRCRN: Boosting Feature Representation Using Frequency Recurrence for Monaural Speech Enhancement}, 
  year={2022},
  pages={9281-9285},
  doi={10.1109/ICASSP43922.2022.9747578}}
```

[2]

```BibTeX
@INPROCEEDINGS{9747230,
  author={Dubey, Harishchandra and Gopal, Vishak and Cutler, Ross and Aazami, Ashkan and Matusevych, Sergiy and Braun, Sebastian and Eskimez, Sefik Emre and Thakker, Manthan and Yoshioka, Takuya and Gamper, Hannes and Aichner, Robert},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Icassp 2022 Deep Noise Suppression Challenge}, 
  year={2022},
  pages={9271-9275},
  doi={10.1109/ICASSP43922.2022.9747230}}
```
