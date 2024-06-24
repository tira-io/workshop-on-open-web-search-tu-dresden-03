#!/usr/bin/env python3
from tira.rest_api_client import Client
from pathlib import Path
from tqdm import tqdm

tira = Client()
approach = 'ir-lab-sose-2024/tu-dresden-03/qe-gpt3.5-cot'
dataset_ids = ['ir-acl-anthology-20240504-training', 'ir-acl-anthology-topics-leipzig-20240423-test', 'ir-acl-anthology-topics-koeln-20240614-test', 'ir-acl-anthology-topics-augsburg-20240525_0-test']

for dataset_id in tqdm(dataset_ids):
    run_file = Path('generated') / 'chain-of-thoughts'/ 'gpt' / f'{dataset_id}.jsonl'
    tira.upload_run(approach=approach, dataset_id=dataset_id, file_path=run_file);

