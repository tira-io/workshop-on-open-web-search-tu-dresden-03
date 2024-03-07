from openai import OpenAI
from models.layout import Layout
from util.utility import save_query
from tqdm import tqdm


class ChatGPTWrapper(Layout):

    def __init__(self, max_len, temperature, name):
        # TODO Remove API KEY
        super().__init__(name)
        self.client = OpenAI(api_key='...')
        self.max_len = max_len
        self.temperature = temperature

    def process_query(self, prompt):
        response = self.client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": f"{prompt}"},
            ],
            max_tokens=self.max_len,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def chain_of_thoughts(self, queries, dset_name):
        def add_context(q):
            return (f'Answer the following query:'
                    f''
                    f'{q}'
                    f''
                    f'Give the rationale before answering.')

        for query in tqdm(queries):
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

        for query in tqdm(queries):
            prompt = add_context(query.default_text())
            response = self.process_query(prompt)
            save_query(exp_name="similar-queries-fs", model_name=self.name, dset_name=dset_name, query=query, response=response)

    def similar_queries_zs(self, queries, dset_name):
        def add_context(q):
            return (f'Suggest 5 queries that are similar to the following query:'
                    f''
                    f'Query: {q}')

        for query in tqdm(queries):
            prompt = add_context(query.default_text())
            response = self.process_query(prompt)
            save_query(exp_name="similar-queries-zs", model_name=self.name, dset_name=dset_name, query=query, response=response)
