<div align="center">
<h1>
  星辰大模型(52B)
</h1>
</div>




# 目录
- [模型介绍](#模型介绍)
- [数据开源](#数据开源)  
- [效果评测](#效果评测)
- [模型推理和部署](#模型推理和部署)
- [模型微调](#模型微调)
- [声明、协议、引用](#声明协议引用)

# 最新动态
- 7.16 [支持vllm推理](https://github.com/Tele-AI/TeleChat-52B/tree/main/vllm_inf)
- 5.16 开源52B版本chat模型

# 模型介绍
### 星辰大模型（52B）
- 星辰大模型（52B）是一款开源多语言大模型，其模型基座使用高质量中英文数据、更优数据配比，采用课程学习方式进行训练。
- 我们开源了使用星辰语义大模型52B基座微调的对话模型，以及基于Deepspeed的微调代码和huggingface推理代码。
- 星辰大模型（52B）在模型评测中取得了领先的效果，在榜单评测上超过LLaMA-2-70B-Chat，与Qwen-72B-chat可比；通用对话性能已经超过GPT-3.5-Turbo。

### 模型结构

我们采用标准的 `Decoder-only` 结构设计了 **TeleChat** 模型，并在模型维度做了如下的一些改进：

- **位置编码**：我们使用 [Rotary Embedding](https://arxiv.org/pdf/2104.09864.pdf) 的位置编码方法，该方法将相对位置信息依赖集成到 self-attention 中，并且具有较好的位置外推性。Rotary Embedding还可以较好地与Flash-Attention v2 配合使用，将模型的训练速度提升约20%。
- **激活函数**：我们使用 [SwiGLU](https://arxiv.org/pdf/2002.05202.pdf) 激活函数来替代GELU激活函数。
- **层标准化**: 基于 [RMSNorm](https://arxiv.org/abs/1910.07467) 的 Pre-Normalization。
- **词嵌入层与输出层解耦**：我们将星辰52B的词嵌入层和输出lm head层参数分开，有助于增强训练稳定性和收敛性。


|         | layer_num | hidden_size | ffn_hidden_size | head_num | tie_word_embeddings |
| ------- | --------- | ----------- | --------------- | -------- | ------------------- |
| 星辰52B | 64        | 8192        | 21824           | 64       | 否                  |

---

我们开源的星辰52B模型：
- 支持deepspeed微调，开源了基于deepspeed的训练代码，支持Zero并行显存优化，同时集成了FlashAttention2
- 多轮能力支持。开源了多轮数据构建方式，针对多轮模型训练集成了针对多轮的mask loss训练方式，更好的聚焦多轮答案，提升问答效果。


本次发布版本和下载链接见下表：

| 模型版本 | 下载链接                                                                  |
| -------- |-----------------------------------------------------------------------|
| 52B-FP16 | [TeleChat-52B-FP16](https://modelscope.cn/models/TeleAI/TeleChat-52B/files) |

**镜像下载**（需修改）
为了便于大家快速上手，我们提供了可运行的环境镜像，下载地址：[镜像下载](https://cloud.189.cn/web/share?code=vQFJRf7JBfmq) （访问码：ona6）



# 效果评测
星辰52B模型相比同规模模型在评测效果方面也有较好的表现，我们的评测集涵盖了包括MMLU、AGIEval、CMMLU、 GSM8K、MATH、HumanEval 等数据集，评测能力包括了自然语言理解、知识、数学计算和推理、代码生成等

## 评测集介绍

### 通用能力

- MMLU 数据集是一个全面的英文评测数据集，涵盖了 57 个学科，包括人文学科、社会科学、自然科学、初等数学、美国历史、计算机科学、法律等等。
- CMMLU 数据集同样是一个全面的中文评估测试集，涵盖了从基础学科到高级专业水平的67个主题。
- AGIEval 数据集是一个专门为评估基础模型在难度较高的标准化考试（如大学入学考试、法学院入学考试、数学竞赛和律师资格考试）的语境中而设计的基准测试，包括中文试题和英文试题。

### 推理和代码能力

- GSM8K 数据集包含了8.5K高质量的小学数学题，能够评估语言模型在数学推理能力上的表现，我们利用[官方](https://github.com/openai/grade-school-math)的评测方案在test集上进行了4-shot测试。

- MATH 数据集包含了12.5K具有挑战性的高中数学竞赛题，难度较大，对语言模型的推理能力要求较高，基于[官方](https://github.com/hendrycks/math)的评测方案，我们在test集上进行了4-shot测试。

- HumanEval 数据集是一个由openai提供的代码能力测试数据集，它由 164 个编程问题组成，要求根据给定的问题和代码模板，生成正确的代码片段，我们利用[官方](https://github.com/openai/human-eval)评测方案在test集上进行了zero-shot测试。



## 评测结果如下

| Model            |   MMLU   |   CMMLU   |  AGIEval  |  GSM8K   |   MATH   | HumanEval |   BBH    | HellaSwag |
| :--------------- | :------: | :-------: | :-------: | :------: | :------: | :-------: | :------: | :-------: |
|                  |  5-shot  |  5-shot   | zero-shot |  4-shot  |  4-shot  | zero-shot |  3-shot  | zero-shot |
| LLaMA-2-70B-Chat |   63.8   |   43.3    |   37.9    |   59.3   |   10.4   |   32.3    |   60.8   |   80.6    |
| Qwen-72B-chat    |    74    |   81.4    |   58.5    |   67.4   |   31.8   |   49.4    |    68    |   84.7    |
| 星辰7B-chat      |   60.5   |   64.3    |   46.8    |   36.7   |   10.3   |   20.1    |   19.5   |   36.7    |
| 星辰12B-chat     |   73.3   |   74.2    |   51.7    |   57.2   |   16.0   |   22.0    |   52.2   |   71.5    |
| **星辰52B-chat** | **76.6** | **73.79** | **61.1**  | **63.5** | **13.5** | **36.6**  | **60.3** | **86.3**  |

说明：榜单均基于[OpenCompass](https://github.com/open-compass/OpenCompass/)平台提供的评测方法进行评估，而对于对比模型，我们同时参考了官方汇报结果和OpenCompass结果。

### 对话能力评测

为了评价模型的对话能力，研发团队建立了包含2500+单轮、多轮对话交互的内部评测系统，涵盖闲聊问答、专业知识、翻译、逻辑思维、长文写作、幻觉测试、安全测试、角色扮演、任务执行、数学能力等多个维度，并使用Judge模型基于详细的评价指标文档进行自动打分。在当前评测数据上，星辰52B模型的综合平均得分为83.8，高于GPT-3.5-Turbo的82.3。这一结果表明，星辰52B模型能较好地支持下游任务应用。



# 模型推理和部署
### 模型推理
当前模型支持fp16精度推理，适配4卡40G A100进行推理。具体推理操作请参考`infer.py`文件，该文件中有单轮和多轮的推理示例。

**模型推理方法示范**
```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
PATH = "/path/to/TeleChat-52B-chat"
tokenizer = AutoTokenizer.from_pretrained(PATH, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(PATH,torch_dtype=torch.bfloat16,device_map='auto',trust_remote_code=True)
question = "你作为一名气候保护协会的会员，你准备写一篇全球气候变化的新闻报告，要求体现出全球气候变化以前与现在情况的对比，字数要求1000字。"
generate_config = GenerationConfig.from_pretrained(PATH)
answer = model.chat(tokenizer,question, history_input_list = [], history_output_list = [],generation_config = generate_config)
print("machine:",answer)
```



# 模型微调

以下是一些性能测试供参考。

全参微调deepspeed版本，8机64卡A100-40G，训练速度参考（ samples/s）

| 模型大小 |   NVIDIA卡型号   | 最长训练长度 |                  参数设置                    |
|:----:|:-------------:|:------:|:--------------------------------------------: |
| 52B  | 8机64卡A100-40G |  4096  | flash-attn开启，zero-3，gradient-checkpointing |
## 数据处理
为了方便数据配比，解耦了数据处理和模型训练，数据权重配比文件如**data.json**所示，json字典中key为读取数据的路径，value为训练时数据的权重。单轮、多轮数据格式如样例数据所示
```shell
{
  "datas/single_turn_example.jsonl": 2.0,
  "datas/multi_turn_example.jsonl": 1.0
}
```
运行**process_data.py**即可将文件处理成tokens，并保存。其中**data_output_path/train_data_{i}.pt**保存处理后的文件，**i的范围是0~num_workers**。训练时会加载路径下所有**train_data_{i}.pt**文件

* 数据通过**data_path**读取，最终拼接生成**num_samples**个**max_seq_len**长度的sample进行训练。如样例所示，假设**datas/single_turn_example.jsonl**和**datas/multi_turn_example.jsonl**各有1000条samples，配比过后数据池中则总共包含3000条samples。在数据拼接过程中，程序会不断遍历数据池，尽可能将数据拼接到4096长度（不够就左padding），直至生成到num_samples的个数。因此，每个sample中会包含多条拼接而成的数据。
* process_method选择**single**或**multiple**单进程或多进程处理数据。

```python
python -u process_data.py \
   --data_path data.json \ # 数据配比文件路径
   --tokenizer_path $MODEL_PATH \ # 模型/tokenzier路径
   --data_output_path $DATA_OUTPUT_PATH \ # 处理后数据保存地址
   --max_seq_len $MAX_LEN \ # 数据长度
   --num_samples $NUM_SAMPLES \ # 最终生成拼接后的数据数量
   --num_workers 10 \ # 多进程个数
   --process_method multiple \ # 多进程&单进程处理
   --seed 42
```


## 模型微调

### 全量训练

可运行**run_telechat_52b.sh**脚本

多机训练需要给出免密互连hostfile，如下所示，node1、node2、node3、node4是节点名称，slots代表每个节点的卡数
```shell
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
```

```shell
bash run_telechat_52b.sh
```

### Lora训练

可运行**run_telechat_52b_lora.sh**脚本

```shell
bash run_telechat_52b_lora.sh
```


# 声明、协议、引用
### 声明
我们在此声明，不要使用TeleChat模型及其衍生模型进行任何危害国家社会安全或违法的活动。同时，我们也要求使用者不要将TeleChat模型用于没有安全审查和备案的互联网服务。我们希望所有使用者遵守上述原则，确保科技发展在合法合规的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用TeleChat开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

### 协议
社区使用 TeleChat 模型需要遵循《[TeleChat模型社区许可协议](./TeleChat模型社区许可协议.pdf)》。TeleChat模型支持商业用途，如果您计划将 TeleChat 模型或其衍生品用于商业目的，您需要通过以下联系邮箱 tele_ai@chinatelecom.cn，提交《TeleChat模型社区许可协议》要求的申请材料。审核通过后，将特此授予您一个非排他性、全球性、不可转让、不可再许可、可撤销的商用版权许可。

### 引用
如需引用我们的工作，请使用如下 reference:
```
@misc{wang2024telechat,
      title={TeleChat Technical Report}, 
      author={Zihan Wang and Xinzhang Liu and Shixuan Liu and Yitong Yao and Yuyao Huang and Zhongjiang He and Xuelong Li and Yongxiang Li and Zhonghao Che and Zhaoxi Zhang and Yan Wang and Xin Wang and Luwen Pu and Huihan Xu and Ruiyu Fang and Yu Zhao and Jie Zhang and Xiaomeng Huang and Zhilong Lu and Jiaxin Peng and Wenjun Zheng and Shiquan Wang and Bingkai Yang and Xuewei he and Zhuoru Jiang and Qiyi Xie and Yanhan Zhang and Zhongqiu Li and Lingling Shi and Weiwei Fu and Yin Zhang and Zilu Huang and Sishi Xiong and Yuxiang Zhang and Chao Wang and Shuangyong Song},
      year={2024},
      eprint={2401.03804},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{li2024teleflm,
      title={Tele-FLM Technical Report}, 
      author={Xiang Li and Yiqun Yao and Xin Jiang and Xuezhi Fang and Chao Wang and Xinzhang Liu and Zihan Wang and Yu Zhao and Xin Wang and Yuyao Huang and Shuangyong Song and Yongxiang Li and Zheng Zhang and Bo Zhao and Aixin Sun and Yequan Wang and Zhongjiang He and Zhongyuan Wang and Xuelong Li and Tiejun Huang},
      year={2024},
      eprint={2404.16645},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
