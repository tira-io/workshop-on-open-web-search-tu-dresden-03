#!/usr/bin/env python

import argparse
from experiments.experiments import ChainOfThoughts, SimilarQueriesFS, SimilarQueriesZS
from models.flan_ul2 import FlanUL2Wrapper
from models.llama import Llama2Wrapper
from models.chatgpt import ChatGPTWrapper
from util.utility import get_all_datasets, get_all_all_datasets, get_by_priority, split_list
from huggingface_hub import snapshot_download
from tira.third_party_integrations import ir_datasets
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-name-or-path', type=str, help='The model name or path to run.', required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--input-dataset', type=str, required=True)
    parser.add_argument('--expansion-type', type=str, help='The expansion to run.', required=True, choices=['chain-of-thought', 'similar-queries-few-shot', 'similar-queries-zero-shot'])

    return parser.parse_args()

if __name__ == '__main__':
    flan_model, llama_model, chatgpt_model = None, None, None
    args = parse_args()

    model_name_or_path = args.model_name_or_path
    model_name_or_path = snapshot_download(args.model_name_or_path, local_files_only=True)

    dset_list = [args.input_dataset]
    dataset = ir_datasets.load(args.input_dataset)
    print(f'Process {len(list(dataset.queries_iter()))} queries from "{dataset}".')

    if 'lama' in args.model_name_or_path:
        llama_model = Llama2Wrapper(min_len=10, max_len=200, temperature=1.1, name="llama", model_name_or_path=model_name_or_path, tokenizer_name_or_path=model_name_or_path)
        print('Use llama model', model_name_or_path)
    elif 'flan' in args.model_name_or_path:
        flan_model = FlanUL2Wrapper(min_len=10, max_len=200, temperature=0.5, name="flan-ul2", model_name_or_path=model_name_or_path, tokenizer_name_or_path=model_name_or_path)
        print('Use flan model', model_name_or_path)

    #chatgpt_model = ChatGPTWrapper(max_len=200, temperature=0.5, name="gpt")


    chain_of_thoughts = ChainOfThoughts(long_name="chain-of-thoughts", short_name="CoT", flan=flan_model, llama=llama_model, gpt=chatgpt_model, dsets=dset_list)
    
    similar_queries_fs = SimilarQueriesFS(long_name="similar-queries-few-shot", short_name="Q2E/FS", flan=flan_model, llama=llama_model, gpt=chatgpt_model, dsets=dset_list)

    similar_queries_zs = SimilarQueriesZS(long_name="similar-queries-zero-shot", short_name="Q2E/ZS", flan=flan_model, llama=llama_model, gpt=chatgpt_model, dsets=dset_list)

    results = []

    if args.expansion_type == 'chain-of-thought':
        results = list(chain_of_thoughts.run())
    elif args.expansion_type == 'similar-queries-few-shot':
        results = list(similar_queries_fs.run())
    elif args.expansion_type == 'similar-queries-zero-shot':
        results = list(similar_queries_zs.run())

    with open(args.output_file, 'w') as f:
        for i in results:
            f.write(json.dumps(i) + '\n')

