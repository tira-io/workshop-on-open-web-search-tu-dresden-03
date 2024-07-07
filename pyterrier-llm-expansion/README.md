
docker build -t llm-expansion .

tira-run \
    --input-dataset reneuir-2024/tiny-sample-20231030-training \
    --image llm-expansion \
    --push true \
    --tira-vm-id tu-dresden-03 \
    --command '/app/retrieve-with-llm-expanded-query.py $inputDataset BM25 --repeat-expanded-query 1 --repeat-original-query 5 --expansion-name query-expansion-flan-ul2-chain-of-thoughts'

