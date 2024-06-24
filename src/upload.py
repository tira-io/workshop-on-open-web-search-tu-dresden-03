#!/usr/bin/env python3
from tira.rest_api_client import Client
from pathlib import Path
from tqdm import tqdm
import pandas as pd

tira = Client()
approach = 'ir-lab-sose-2024/tu-dresden-03/qe-llama-sq-fs'
dataset_ids = ['ir-acl-anthology-20240504-training', 'ir-acl-anthology-topics-leipzig-20240423-test', 'ir-acl-anthology-topics-koeln-20240614-test', 'ir-acl-anthology-topics-augsburg-20240525_0-test']

for dataset_id in tqdm(dataset_ids):
    path = Path('generated') / 'similar-queries-fs'/ 'llama'
    run_file = path / f'{dataset_id}.jsonl'

    df = pd.read_json(run_file, lines=True)
    df['qid'] = df['query-id'].astype(str)
    df['query'] = df['response'].astype(str)
    del df['query-id']
    del df['response']
    del df['query-text']
    del df['metadata']
    run_file = path / f'{dataset_id}.jsonl.gz'
    df.to_json(run_file, lines=True, orient='records')

    tira.upload_run(approach=approach, dataset_id=dataset_id, file_path=run_file);

