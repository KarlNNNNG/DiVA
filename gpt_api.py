from typing import List
import openai
import os
from time import sleep
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

openai.api_key = ""
proxy = {
'http': 'http://localhost:7890',
'https': 'http://localhost:7890'
}

import requests
import json

# class Demo():
#     def __init__(self):
#         url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=gPKQ7QafWARjZHiE0JQlC9KN&client_secret=W25pcNSkUet8lz9H1EC0YWQOF7P7lCPx"
    
#         payload = json.dumps("")
#         headers = {
#             'Content-Type': 'application/json',
#             'Accept': 'application/json'
#         }
    
#         response = requests.request("POST", url, headers=headers, data=payload)
#         self.access_token = response.json().get("access_token")
#         self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_13b?access_token=" + self.access_token

#     def get_multiple_sample(self, prompt_list: List[str]):
#         payload = json.dumps({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt_list
#             }
#         ]
#         })
#         headers = {
#             'Content-Type': 'application/json'
#         }
        
#         response = requests.request("POST", self.url, headers=headers, data=payload)

#         result = response.json().get("result")
#         if result == None:
#             result = "None"
#         print(result)
#         return result, 1.0

openai.proxy = proxy

class Demo(object):
    def __init__(self):
        self.client = openai.OpenAI()
        OPENAI_API_KEY = ""
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        

    def get_multiple_sample(self, prompt_list: List[str]):
        if len(prompt_list) == 1:
            message = [
                    {"role": "system", "content": prompt_list[0]},
                ]
        elif len(prompt_list)==2:     
            message = [
                {"role": "system", "content": prompt_list[0]},
                {"role": "user", "content": prompt_list[1]}
            ]      
        # messages=[
        #             {"role": "system", "content": prompt_list[0]},
        #         ],

        content = "None"
        while content == "None":
            try:
                completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=message,
                logprobs=True,
                temperature=0.9
                )
                content = completion.choices[0].message.content
                print(completion.choices[0].message.content)
            except:
                print("Retry Connect OpenAI")
        

        # response = openai.completions.create(
        #     model=self.engine,
        #     prompt=prompt_list,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     top_p=self.top_p,
        #     frequency_penalty=self.frequency_penalty,
        #     presence_penalty=self.presence_penalty,
        #     best_of=self.best_of,
        #     logprobs=self.logprobs
        # )
        return content


class Llama(object):
    def __init__(self):
        model_id = "D:\\Projects\\llama3-main\\nvidiaLlama3-ChatQA-8B"
        self.max_tokens = 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")

    def get_multiple_sample(self, prompt_list: List[str]):
        tokenized_prompt = self.tokenizer(self.tokenizer.bos_token + prompt_list, return_tensors="pt").to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators)

        response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True), 1.0


class LLM(object):
    def __init__(self):
        model_id = "D:\\Projects\\GoLLIE"
        self.max_tokens = 1024
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")

    def get_multiple_sample(self, prompt_list: List[str]):
        tokenized_prompt = self.tokenizer(self.tokenizer.bos_token + prompt_list[0], return_tensors="pt").to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=self.max_tokens, eos_token_id=terminators)

        response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)



article_text = """The U.S. Bureau of Ocean Energy Management (BOEM) has finalized an extensive environmental review concerning wind development activities across six lease areas in the New York Bight, covering more than 488,000 acres off the coasts of New York and New Jersey. This initiative is projected to yield up to 7 gigawatts (GW) of offshore wind energy—enough to supply power to approximately two million households.BOEM Director Elizabeth Klein emphasized the importance of stakeholder engagement, stating, "We have gathered input from Tribes, government agencies, local communities, and ocean users. This feedback has been invaluable, and our regional approach will set a strong foundation for future environmental assessments of proposed offshore wind projects in the New York Bight."The review follows a record-setting auction in February 2022, where BOEM raised over $4.3 billion for leasing rights in these areas, marking the highest revenue generated from any U.S. offshore energy lease sale to date.The Programmatic Environmental Impact Statement (PEIS) prepared by BOEM evaluates the environmental implications of proposed offshore wind activities in the region. It outlines strategies for avoiding, minimizing, mitigating, and monitoring impacts—key measures that will be incorporated into the approval process for specific projects proposed by lessees. This comprehensive regional analysis marks a first for BOEM, encompassing multiple lease areas in offshore renewable energy development.From 2022 to 2024, BOEM facilitated public engagement through five public meetings and eight regional environmental justice forums, with support from the Inflation Reduction Act. This outreach aimed to gather insights on vital resources, environmental concerns, and the proposed AMMM measures. The agency received a substantial response, with 1,568 unique comments from 560 submissions, which informed the Final PEIS.Under the Biden-Harris administration, the Department of the Interior has authorized over 15 GW of clean energy from ten offshore wind projects, enough to power nearly 5.25 million homes. The administration has also conducted five offshore wind lease auctions, including the groundbreaking sale in the New York Bight and the inaugural sales in the Pacific and Gulf of Mexico.The “Notice of Availability of a Final Programmatic Environmental Impact Statement for Expected Wind Energy Development in the New York Bight” is set to be published in the Federal Register on October 25, 2024, marking a significant step forward in the expansion of renewable energy infrastructure in the region."""