#!/usr/bin/env python3
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
import pyterrier as pt
from tira.rest_api_client import Client

import argparse
parser = argparse.ArgumentParser(description='Retrieve with PyTerrier (For the ReNeuIR Workshop).')
parser.add_argument('dataset', type=str)
parser.add_argument('wmodel', type=str)
parser.add_argument('--repeat-original-query', default=5, type=int)
parser.add_argument('--repeat-expanded-query', default=1, type=int)
parser.add_argument('--expansion-name', required=True, choices=['query-expansion-flan-ul2-chain-of-thoughts', 'query-expansion-flan-ul2-similar-queries-few-shot',
                                                                'query-expansion-flan-ul2-similar-queries-zero-shot', 'query-expansion-llama2-chat-chain-of-thoughts',
                                                                'query-expansion-llama2-chat-similar-queries-zero-shot', 'query-expansion-llama2-chat-similar-queries-few-shot'
                                                                ])

args = parser.parse_args()

ensure_pyterrier_is_loaded()
tira = Client()

tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def pt_tokenise(text):
    return ' '.join(tokeniser.getTokens(text))

# In the TIRA sandbox, this is the injected ir_dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
pt_dataset = pt.get_dataset(f'irds:{args.dataset}')

index = tira.pt.index('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', pt_dataset)

expansions = {}
expansions['query-expansion-flan-ul2-chain-of-thoughts'] = tira.pt.transform_queries('reneuir-2024/tu-dresden-03/query-expansion-flan-ul2-chain-of-thoughts', pt_dataset)
expansions['query-expansion-flan-ul2-similar-queries-few-shot'] = tira.pt.transform_queries('reneuir-2024/tu-dresden-03/query-expansion-flan-ul2-similar-queries-few-shot', pt_dataset)
expansions['query-expansion-flan-ul2-similar-queries-zero-shot'] = tira.pt.transform_queries('reneuir-2024/tu-dresden-03/query-expansion-flan-ul2-similar-queries-zero-shot', pt_dataset)
expansions['query-expansion-llama2-chat-chain-of-thoughts'] = tira.pt.transform_queries('reneuir-2024/tu-dresden-03/query-expansion-llama2-chat-chain-of-thoughts', pt_dataset)
expansions['query-expansion-llama2-chat-similar-queries-zero-shot'] = tira.pt.transform_queries('reneuir-2024/tu-dresden-03/query-expansion-llama2-chat-similar-queries-zero-shot', pt_dataset)
expansions['query-expansion-llama2-chat-similar-queries-few-shot'] = tira.pt.transform_queries('reneuir-2024/tu-dresden-03/query-expansion-llama2-chat-similar-queries-few-shot', pt_dataset)

print('Retrieve with ' + args.wmodel + ' on "{args.expansion_name}" expansions.')
topics = pt_dataset.get_topics("title")
expansions = expansions[args.expansion_name]
topics = expansions(topics)
expansion_name = '-'.join(args.expansion_name.split('-')[4:]) + '-expansion'
topics[expansion_name] = topics[expansion_name].apply(pt_tokenise)
topics['query'] = topics.apply(lambda i: ' '.join([i['query']]*args.repeat_original_query) + ' ' + ' '.join([i[expansion_name]]*args.repeat_expanded_query), axis=1)
del topics[expansion_name]
print('Show some example queries')
for _, i in topics.head(5).iterrows():
    print(i.to_dict())

bm25 = pt.BatchRetrieve(index, wmodel=args.wmodel, verbose=True)

print('Create run')
run = bm25(topics)
print('Done, run was created')

# In the TIRA sandbox, this uses an environment variable to persist the run to the correct output directory
persist_and_normalize_run(run, args.wmodel + '-default_weights')    
