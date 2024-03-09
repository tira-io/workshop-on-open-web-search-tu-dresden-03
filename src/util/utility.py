import jsonlines
import os
import json


def save_query(exp_name, model_name, dset_name, query, response):
    path = f'generated/{exp_name}/{model_name}'

    if not os.path.exists(path):
        os.makedirs(path)

    with jsonlines.open(f'{path}/{dset_name}.jsonl', mode='a') as writer:
        elem = {
            "query-id": query.query_id,
            "query-text": query.default_text(),
            "response": response,
            "metadata": {"model": model_name}
        }
        writer.write(elem)


def get_idx_of_last_query(exp_name, model_name, dset_name):
    return len(read_queries(exp_name, model_name, dset_name))


def read_queries(exp_name, model_name, dset_name):
    path = f'generated/{exp_name}/{model_name}/{dset_name}.jsonl'
    try:
        with open(path) as file:
            data = [line.rstrip() for line in file]
    except FileNotFoundError:
        return []

    return [json.loads(entry) for entry in data]


def get_all_datasets():
    return [
        'msmarco-passage-trec-dl-2019-judged-20230107-training',
        'msmarco-passage-trec-dl-2020-judged-20230107-training', 'longeval-long-september-20230513-training',
        'longeval-short-july-20230513-training', 'longeval-train-20230513-training',
        'longeval-heldout-20230513-training', 'antique-test-20230107-training',
        'argsme-touche-2021-task-1-20230209-training', 'argsme-touche-2020-task-1-20230209-training',
        'nfcorpus-test-20230107-training', 'trec-tip-of-the-tongue-dev-20230607-training', 'vaswani-20230107-training',
        'cord19-fulltext-trec-covid-20230107-training', 'medline-2004-trec-genomics-2004-20230107-training',
        'medline-2004-trec-genomics-2005-20230107-training', 'medline-2017-trec-pm-2017-20230211-training',
        'medline-2017-trec-pm-2018-20230211-training',
        'cranfield-20230107-training'
    ]

def get_all_all_datasets():
    return [
        'antique-test-20230107-training', 'argsme-touche-2021-task-1-20230209-training', 'argsme-touche-2020-task-1-20230209-training',
        'clueweb09-en-trec-web-2009-20230107-training', 'clueweb09-en-trec-web-2010-20230107-training', 'clueweb09-en-trec-web-2011-20230107-training',
        'clueweb09-en-trec-web-2012-20230107-training', 'clueweb12-touche-2020-task-2-20230209-training', 'clueweb12-touche-2021-task-2-20230209-training',
        'clueweb12-trec-misinfo-2019-20240214-training', 'clueweb12-trec-web-2013-20230107-training', 'clueweb12-trec-web-2014-20230107-training',
        'cord19-fulltext-trec-covid-20230107-training', 'cranfield-20230107-training', 'disks45-nocr-trec-robust-2004-20230209-training',
        'disks45-nocr-trec7-20230209-training', 'disks45-nocr-trec8-20230209-training', 'gov-trec-web-2002-20230209-training',
        'gov-trec-web-2003-20230209-training', 'gov-trec-web-2004-20230209-training', 'gov2-trec-tb-2004-20230209-training',
        'gov2-trec-tb-2005-20230209-training', 'gov2-trec-tb-2006-20230209-training', 'longeval-heldout-20230513-training',
        'longeval-long-september-20230513-training', 'longeval-short-july-20230513-training', 'longeval-train-20230513-training',
        'medline-2004-trec-genomics-2004-20230107-training', 'medline-2004-trec-genomics-2005-20230107-training', 'medline-2017-trec-pm-2017-20230211-training',
        'medline-2017-trec-pm-2018-20230211-training', 'msmarco-passage-trec-dl-2019-judged-20230107-training', 'msmarco-passage-trec-dl-2020-judged-20230107-training',
        'nfcorpus-test-20230107-training', 'trec-tip-of-the-tongue-dev-20230607-training', 'vaswani-20230107-training',
        'wapo-v2-trec-core-2018-20230107-training'
    ]

def get_by_priority(tranche_id):
    return [
        ['msmarco-passage-trec-dl-2019-judged-20230107-training',
         'msmarco-passage-trec-dl-2020-judged-20230107-training', 'longeval-long-september-20230513-training',
         'longeval-short-july-20230513-training', 'longeval-train-20230513-training',
         'longeval-heldout-20230513-training', 'antique-test-20230107-training',
         'argsme-touche-2021-task-1-20230209-training', 'argsme-touche-2020-task-1-20230209-training'],
        ['nfcorpus-test-20230107-training', 'trec-tip-of-the-tongue-dev-20230607-training', 'vaswani-20230107-training',
         'cord19-fulltext-trec-covid-20230107-training', 'medline-2004-trec-genomics-2004-20230107-training',
         'medline-2004-trec-genomics-2005-20230107-training', 'medline-2017-trec-pm-2017-20230211-training',
         'medline-2017-trec-pm-2018-20230211-training'],
        ['cranfield-20230107-training']
    ][tranche_id]


def get_similar(name):
    return next((s for s in get_all_datasets() if name in s), [])


def split_list(lst, parts):
    if parts > len(lst):
        ret = [[lst[i]] for i in range(len(lst))]
        if len(ret) < parts:
            for i in range(parts-len(ret)):
                ret.append([])
#        print("returning " + str(ret))
        return ret

    avg = len(lst) / float(parts)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out
