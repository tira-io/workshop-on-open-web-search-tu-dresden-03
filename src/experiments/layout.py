import pyterrier as pt
import pandas as pd
from tqdm import tqdm

from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
from util.utility import *

tira = Client()
ensure_pyterrier_is_loaded()


class Layout:
    def __init__(self, long_name, short_name, dsets, flan, llama, gpt):
        self.flan = flan
        self.llama = llama
        self.gpt = gpt
        self.dsets = dsets
        self.name = short_name
        self.exp_name = long_name

    def eval_all(self):
        for model_name in [self.flan.name, self.llama.name, self.gpt.name]:
            self.do_evaluation(self.exp_name, model_name)

    def eval(self, model_names):
        for model_name in model_names:
            print(f"Evaluating model {model_name} ...")
            self.do_evaluation(self.exp_name, model_name)

    def do_evaluation(self, exp_name, model_name):
        eval_dfs = []
        for dset_name in tqdm(self.dsets):
            expanded_queries = Layout.get_as_dict(exp_name, model_name, dset_name)
            pt_dataset = pt.get_dataset(f"irds:ir-benchmarks/{dset_name}")

            pt_expand_query = pt.apply.generic(lambda i: Layout.expand_queries_in_dataframe(i, expanded_queries))

            index = Layout.pyterrier_index_from_tira(dset_name)

            bm25 = pt.BatchRetrieve(index, wmodel="BM25")
            bm25_rm3 = bm25 >> pt.rewrite.RM3(index) >> bm25
            bm25_kl = bm25 >> pt.rewrite.KLQueryExpansion(index) >> bm25
            bm25_with_expansion = pt_expand_query >> bm25

            df = pt.Experiment([bm25, bm25_rm3, bm25_kl, bm25_with_expansion], pt_dataset.get_topics('query'),
                               pt_dataset.get_qrels(),
                               ['ndcg_cut.10', 'recall_1000'],
                               names=['BM25', 'BM25+RM3', 'BM25+KL', f'BM25+{self.name}'], verbose=True)
            df['dataset'] = dset_name
            df['model'] = model_name
            eval_dfs.append(df)

        eval_df = pd.concat(eval_dfs)

        path = f'generated/{exp_name}/evaluation'
        if not os.path.exists(path):
            os.makedirs(path)

        eval_df.to_json(f'{path}/eval-{model_name}.jsonl', lines=True, orient='records')

    # Generates a file for every experiment and dataset (3 models * 4 experiments = 12 lines per file)
    def eval_dataset(self, model_names):
        print(f"[Layout::eval_dataset] {self.exp_name} is doing evaluation ...")
        for dset_name in tqdm(self.dsets):
            eval_dfs = []
            for model_name in model_names:
                print(f"\n[Layout::eval_dataset] {self.exp_name} is doing evaluation for {model_name} on dataset {dset_name}")
                expanded_queries = Layout.get_as_dict(self.exp_name, model_name, dset_name)
                pt_dataset = pt.get_dataset(f"irds:ir-benchmarks/{dset_name}")

                pt_expand_query = pt.apply.generic(lambda i: Layout.expand_queries_in_dataframe(i, expanded_queries))

                index = Layout.pyterrier_index_from_tira(dset_name)

                bm25 = pt.BatchRetrieve(index, wmodel="BM25")
                bm25_rm3 = bm25 >> pt.rewrite.RM3(index) >> bm25
                bm25_kl = bm25 >> pt.rewrite.KLQueryExpansion(index) >> bm25
                bm25_with_expansion = pt_expand_query >> bm25

                df = pt.Experiment([bm25, bm25_rm3, bm25_kl, bm25_with_expansion], pt_dataset.get_topics('query'),
                                   pt_dataset.get_qrels(),
                                   ['ndcg_cut.10', 'recall_1000'],
                                   names=['BM25', 'BM25+RM3', 'BM25+KL', f'BM25+{self.name}'], verbose=False)
                df['dataset'] = dset_name
                df['model'] = model_name
                eval_dfs.append(df)

            eval_df = pd.concat(eval_dfs)

            path = f'generated/{self.exp_name}/evaluation/'
            if not os.path.exists(path):
                os.makedirs(path)

            eval_df.to_json(f'{path}/eval-{dset_name}.jsonl', lines=True, orient='records')

    @staticmethod
    def concat(query, reps, llm_output):
        if type(llm_output) is list:
            llm_output = " ".join(llm_output)
        return f'{query} ' * reps + llm_output

    @staticmethod
    def pyterrier_index_from_tira(dataset):
        ret = tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', dataset) + '/index'
        return pt.IndexFactory.of(ret)

    @staticmethod
    def get_as_dict(exp_name, model, dset_name):
        json_res = read_queries(exp_name, model, dset_name)
        tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

        def pt_tokenize(text):
            return ' '.join(tokeniser.getTokens(text))

        expanded_queries = {q['query-id']: pt_tokenize(Layout.concat(q['query-text'], 5, q['response'])) for q in
                            json_res}
        return expanded_queries

    @staticmethod
    def expand_queries_in_dataframe(df, expanded_queries):
        df['text'] = df['qid'].apply(lambda x: expanded_queries[x])
        df['query'] = df['qid'].apply(lambda x: expanded_queries[x])
        return df
