from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

from models.layout import Layout
from util.utility import save_query


class FlanUL2Wrapper(Layout):
    def __init__(self, min_len, max_len, temperature, name):
        super().__init__(name)
        self.min_len = min_len
        self.max_len = max_len
        self.temperature = temperature

        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

    def process_query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        output = self.model.generate(inputs, do_sample=True, min_length=self.min_len, max_length=self.max_len,
                                     temperature=self.temperature)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def chain_of_thoughts(self, queries, dset_name):
        def add_context(q):
            return (f'Answer the following query:'
                    f''
                    f'{q}'
                    f''
                    f'Give the rationale before answering.')

        for query in queries:
            prompt = add_context(query.default_text())
            response = self.process_query(prompt)
            save_query(exp_name="chain-of-thoughts", model_name=self.name, dset_name=dset_name, query=query, response=response)

    def similar_queries_fs(self, queries, dset_name):
        def add_context(q):
            return (f'For every query, suggest a similar query:'
                    f''
                    f'Original query: How to tie a windsor knot?'
                    f'Similar query: Instructions for tying a windsor knot'
                    f''
                    f'Original query: How is the weather tomorrow morning?'
                    f'Similar query: Weather tomorrow morning'
                    f''
                    f'Original query: Simple vegan cooking recipes'
                    f'Similar query: What are some delicious and simple vegan cooking recipes?'
                    f''
                    f'Original Query: {q}'
                    f'Similar Query:')

        for query in queries:
            prompt = add_context(query.default_text())
            response = self.process_query(prompt)
            save_query(exp_name="similar-queries-fs", model_name=self.name, dset_name=dset_name, query=query, response=response)

    def similar_queries_zs(self, queries, dset_name):
        def add_context(q):
            return (f'Suggest 5 queries that are similar to the following query:'
                    f''
                    f'Query: {q}')

        for query in queries:
            prompt = add_context(query.default_text())
            response = self.process_query(prompt)
            save_query(exp_name="similar-queries-zs", model_name=self.name, dset_name=dset_name, query=query, response=response)
