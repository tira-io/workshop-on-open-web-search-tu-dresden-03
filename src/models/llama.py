#!/usr/bin/env python3

import json
import math

from torch import bfloat16
from transformers import LlamaForCausalLM, LlamaTokenizerFast, TextStreamer

from models.layout import Layout
from util.utility import save_query


class Llama2Wrapper(Layout):
    def __init__(self, min_len, max_len, temperature, name, model_name_or_path, tokenizer_name_or_path, **kwargs):
        super().__init__(name)
        self.min_len = min_len
        self.max_len = max_len
        self.temperature = temperature
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name_or_path, use_default_system_prompt=False)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=bfloat16, device_map="auto", load_in_4bit=True)
        # self.sysprompt = "Answer the following query. Be short and concise, 50 words at max. Answer in full sentences."
        # self.prompt = "<s>[INST] <<SYS>>\n" + self.sysprompt + "\n<</SYS>>\n\n"

    @staticmethod
    def make_prompt(sysprompt, prompt):
        return "<s>[INST] <<SYS>>\n" + sysprompt + "\n<</SYS>>\n\n" + prompt.strip() + " [/INST]"

    def tokenize_and_count(self, string):
        model_input = self.tokenizer(string, return_tensors="pt").to("cuda")
        num_tokens = model_input["input_ids"].shape[-1]
        return model_input, num_tokens

    def process_query(self, query, prompttype="zs", show_output=True):
        supported_prompttypes = ["zs", "cot", "fs"]
        if prompttype not in supported_prompttypes:
            print(f"Prompttype {prompttype} is unsupported! Please use one of {supported_prompttypes}.")
            exit(1)
        match prompttype:
            case "zs":
                sysprompt = "Answer the following query. Be short and concise, 50 words at max. Answer in full sentences."
                input = self.make_prompt(sysprompt, query)
            case "cot":
                sysprompt = "Be short and concise, 100 words max. Answer in full sentences, while briefly writing down your steps towards the response."
                input = self.make_prompt(sysprompt, query)
            case "fs":
                sysprompt = "Be short and concise, 100 words max. Answer in full sentences, while briefly writing down your steps towards the response."
                example = (f'For every query, suggest a similar query:'
                    f''
                    f'Original query: How to tie a windsor knot? [/INST]'
                    f'Similar query: Instructions for tying a windsor knot </s>'
                    f''
                    f'<s>[INST] Original query: How is the weather tomorrow morning? [/INST]'
                    f'Similar query: Weather tomorrow morning </s>'
                    f''
                    f'<s>[INST] Original query: Simple vegan cooking recipes [/INST]'
                    f'Similar query: What are some delicious and simple vegan cooking recipes? </s>'
                    f''
                    f'<s>[INST] Original query: {query} [/INST]'
                    f'Similar query:')
                input = self.make_prompt(sysprompt, example)[:-8]

        input, num_input_tokens = self.tokenize_and_count(input)
        output = self.model.generate(**input, max_new_tokens=self.max_len, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, temperature=self.temperature, min_length=self.min_len)
        output = self.tokenizer.decode(output[0][num_input_tokens:], skip_special_tokens=True).strip()

        if show_output:
            print("\nQuery: " + query + "\nResponse: " + output)

        return output

    def process_queries(self, queries, exp_name, dset_name, experiment="cot", show_output=True):
        if show_output:
            for idx, q in enumerate(queries):
                print("\n[" + " " * (math.ceil(math.log10(len(queries)) - 1) - math.floor(math.log10(idx + 1))) + str(idx+1) + "/" + str(len(queries)) + "]", end=" ")
                output = self.process_query(q.text, prompttype=experiment, show_output=show_output)
                save_query(exp_name=exp_name, model_name=self.name, dset_name=dset_name, query=q, response=output)
                yield {"query_id": q.query_id, f"{exp_name}-expansion": output}
        else:
            for q in queries:
                output = self.process_query(q.text, prompttype=experiment, show_output=show_output)
                save_query(exp_name=exp_name, model_name=self.name, dset_name=dset_name, query=q, response=output)
                yield {"query_id": q.query_id, f"{exp_name}-expansion": output}

    def chain_of_thoughts(self, queries, dset_name):
        yield from self.process_queries(queries, exp_name="chain-of-thoughts", dset_name=dset_name, experiment="cot", show_output=False)

    def similar_queries_fs(self, queries, dset_name):
        yield from self.process_queries(queries, exp_name="similar-queries-few-shot", dset_name=dset_name, experiment="fs", show_output=False)

    def similar_queries_zs(self, queries, dset_name):
        yield from self.process_queries(queries, exp_name="similar-queries-zero-shot", dset_name=dset_name, experiment="zs", show_output=False)

