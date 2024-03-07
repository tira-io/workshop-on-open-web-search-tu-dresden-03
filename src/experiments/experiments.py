from experiments.layout import Layout
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
import ir_datasets
from tqdm import tqdm
from util.utility import *

tira = Client()
ensure_pyterrier_is_loaded()


class ChainOfThoughts(Layout):
    def run(self):
        print("[ChainOfThoughts::run] LLMs are working ...")
        for dset_name in tqdm(self.dsets):
            dataset = ir_datasets.load(f'ir-benchmarks/{dset_name}')
            queries = list(dataset.queries_iter())

            if self.flan is not None:
                print(f"[ChainsOfThoughts::run] {self.flan.name} working on {dset_name}")
                idx = get_idx_of_last_query(self.exp_name, self.flan.name, dset_name)
                self.flan.chain_of_thoughts(queries[idx:], dset_name)

            if self.llama is not None:
                print(f"[ChainsOfThoughts::run] {self.llama.name} working on {dset_name}")
                idx = get_idx_of_last_query(self.exp_name, self.llama.name, dset_name)
                self.llama.chain_of_thoughts(queries[idx:], dset_name)

            if self.gpt is not None:
                print(f"[ChainsOfThoughts::run] {self.gpt.name} working on {dset_name}")
                idx = get_idx_of_last_query(self.exp_name, self.gpt.name, dset_name)
                self.gpt.chain_of_thoughts(queries[idx:], dset_name)


class SimilarQueriesFS(Layout):
    def run(self):
        print("[SimilarQueriesFS::run] LLMs are working ...")
        for dset_name in tqdm(self.dsets):
            dataset = ir_datasets.load(f'ir-benchmarks/{dset_name}')
            queries = list(dataset.queries_iter())

            if self.flan is not None:
                print(f"[SimilarQueriesFS::run] {self.flan.name} working on {dset_name}")
                idx = get_idx_of_last_query(self.exp_name, self.flan.name, dset_name)
                self.flan.similar_queries_fs(queries[idx:], dset_name)

            if self.llama is not None:
                print(f"[SimilarQueriesFS::run] {self.llama.name} working on {dset_name}")
                idx = get_idx_of_last_query(self.exp_name, self.llama.name, dset_name)
                self.llama.similar_queries_fs(queries[idx:], dset_name)

            if self.gpt is not None:
                print(f"[SimilarQueriesFS::run] {self.gpt.name} working on {dset_name}")
                idx = get_idx_of_last_query(self.exp_name, self.gpt.name, dset_name)
                self.gpt.similar_queries_fs(queries[idx:], dset_name)


class SimilarQueriesZS(Layout):
    def run(self):
        print("[SimilarQueriesZS::run] LLMs are working ...")
        for dset_name in tqdm(self.dsets):
            dataset = ir_datasets.load(f'ir-benchmarks/{dset_name}')
            queries = list(dataset.queries_iter())

            if self.flan is not None:
                print(f"[SimilarQueriesZS::run] {self.flan.name} working on {dset_name}")
                idx = get_idx_of_last_query(self.exp_name, self.flan.name, dset_name)
                self.flan.similar_queries_zs(queries[idx:], dset_name)

            if self.llama is not None:
                print(f"[SimilarQueriesZS::run] {self.llama.name} working on {dset_name}")
                idx = get_idx_of_last_query(self.exp_name, self.llama.name, dset_name)
                self.llama.similar_queries_zs(queries[idx:], dset_name)

            if self.gpt is not None:
                print(f"[SimilarQueriesZS::run] {self.gpt.name} working on {dset_name}")
                idx = get_idx_of_last_query(self.exp_name, self.gpt.name, dset_name)
                self.gpt.similar_queries_zs(queries[idx:], dset_name)
