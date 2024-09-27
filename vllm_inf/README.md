# TeleChat_52b vllm 推理使用方式

## 下载vllm
```
pip install vllm==0.5.1
```

## vllm 添加telechat

### 将telechat model文件放入
pip show vllm 找到vllm对应位置并进入
```
cd ./vllm/model_executor/models/
```
将此路径下的telechat.py 文件放入以上路径

### 修改init文件
修改同路径下的__init__.py
```
    "StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),
    "TeleChatForCausalLM": ("telechat", "TeleChatForCausalLM"),  #telechat52b
    "ArcticForCausalLM": ("arctic", "ArcticForCausalLM"),
    "XverseForCausalLM": ("xverse", "XverseForCausalLM"),
```
添加以上代码中的TeleChat 一行

### 修改模型文件里的config.json
```
>>> architectures": [
>>>     "TeleChatForCausalLM"
>>>     ]
```

## 启动
按vllm里的方式启动telechat 推理

## 外推
如有外推需求在模型文件中的config.json 中添加
```
>>> "rope_scaling": {
>>>     "factor": 4.0,
>>>     "original_max_position_embeddings": 16384,
>>>     "rope_type": "dynamic"
>>> }
```
在configuration_telechat.py中添加 rope_scaling 参数
在def __init__() 入参中中添加
```
>>> rope_scaling=None
```
在def __init__() 方法中添加
```
self.rope_scaling = rope_scaling
```

#### 示例
```
>>> from vllm import LLM, SamplingParams
>>> import torch
>>> llm = LLM(model="模型路径", trust_remote_code=True, tensor_parallel_size=4, dtype='bfloat16')
>>> prompts = ['<_user>你好<_bot>']
>>> sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
>>> outputs = llm.generate(prompts, sampling_params)
>>> for output in outputs:
>>>     generated_text = output.outputs[0].text
>>>     print(generated_text)
```
