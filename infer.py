import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
PATH = "/PATH/TO/Models" ## 模型路徑
tokenizer = AutoTokenizer.from_pretrained(PATH, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(PATH,
                                             torch_dtype=torch.bfloat16,
                                             device_map='auto',
                                             trust_remote_code=True)
model.eval()


if __name__ == '__main__':
    questions = ["世界上最潮湿的地方是哪里？",
                 "你作为一名气候保护协会的会员，你准备写一篇全球气候变化的新闻报告，要求体现出全球气候变化以前与现在情况的对比，字数要求1000字。"]
    #### single turn example
    generate_config = GenerationConfig.from_pretrained(PATH)
    for question in questions:
        answer = model.chat(tokenizer,question, history_input_list = [], history_output_list = [],generation_config = generate_config)
        print("machine:",answer)
    ### multi turn example
    for question in questions:
        answer = model.chat(tokenizer,question, history_input_list=["愚人节是在每一年的哪一天？","世界贸易组织的缩写是?"], history_output_list=["愚人节是每年的4月1日。","世界贸易组织（WTO）的缩写是WTO。"],generation_config = generate_config)
        print("machine:",answer)
