#!/usr/bin/env python

import json
import os
import matplotlib
from matplotlib import pyplot as plt
from statistics import mean, stdev

# colors
orange = "#ffa500"
violet = "#868ad1"
darkorange = "#f47264"
turquoise = "#84cbc5"
purple = "#7030a0"
navyblue = "#1f75bb"

def load_evals():
    cot_evals = set(os.listdir("generated/chain-of-thoughts/evaluation"))
    fs_evals = set(os.listdir("generated/similar-queries-fs/evaluation"))
    zs_evals = set(os.listdir("generated/similar-queries-zs/evaluation"))

    # check if all dirs contain evals for the same datasets and skip datasets for which not all evals are present
    if not (cot_evals == fs_evals and cot_evals == zs_evals):
        print("There are evaluation data mismatches!")
        all_dss = cot_evals.union(fs_evals, zs_evals)
        eval_dss = cot_evals.intersection(fs_evals, zs_evals)
        print(f"Evaluating for {len(eval_dss)}/{len(all_dss)}, skipping {all_dss - eval_dss}")
    else:
        eval_dss = cot_evals

    data = {}
    for ds in eval_dss:
#        dshat = ds[5:-6]
        dshat = ds[5:-24]
        data[dshat] = {"baselines": {}}
        for exp in ["chain-of-thoughts", "similar-queries-fs", "similar-queries-zs"]:
            data[dshat][exp] = {}
            with open(f"generated/{exp}/evaluation/{ds}", "r") as f:
                tmpdata = [json.loads(line) for line in f.readlines()]
                for line in tmpdata:
                    # baselines are always identical between models and experiments
                    if line["name"] in ["BM25", "BM25+RM3", "BM25+KL"]:
                        data[dshat]["baselines"][line["name"]] = {}
                        data[dshat]["baselines"][line["name"]]["NDCG@10"] = line["ndcg_cut.10"]
                        data[dshat]["baselines"][line["name"]]["Recall@1000"] = line["recall_1000"]
                    else:
                        data[dshat][exp][line["model"]] = {}
                        data[dshat][exp][line["model"]]["NDCG@10"] = line["ndcg_cut.10"]
                        data[dshat][exp][line["model"]]["Recall@1000"] = line["recall_1000"]

    return data

def barchart(data, name, score):
    experiments = ["chain-of-thoughts", "similar-queries-fs", "similar-queries-zs"]
#    plt.xkcd()
    matplotlib.rcParams.update({'font.size': 13})
    plt.figure(figsize=(8, 6))

    barwidth = 0.9
    # baseline
    xs_baseline = [1,2,3]
    ys_baseline = [data[name]["baselines"][m][score] for m in ["BM25", "BM25+RM3", "BM25+KL"]]
    plt.bar(xs_baseline, ys_baseline, barwidth, label=["BM25", "BM25+RM3", "BM25+KL"], color=[orange, violet, darkorange])
    plt.hlines(ys_baseline, [x-barwidth/2 for x in xs_baseline], 15+barwidth/2, colors=[orange, violet, darkorange], linestyles="dashed")
    # flan
    xs_flan = [5,9,13]
    ys_flan = [data[name][exp]["flan-ul2"][score] for exp in experiments]
    plt.bar(xs_flan, ys_flan, barwidth, label="flan", color=turquoise)
    # llama
    xs_llama = [6,10,14]
    ys_llama = [data[name][exp]["llama"][score] for exp in experiments]
    plt.bar(xs_llama, ys_llama, barwidth, label="llama", color=purple)
    # gpt
    xs_gpt = [7,11,15]
    ys_gpt = [data[name][exp]["gpt"][score] for exp in experiments]
    plt.bar(xs_gpt, ys_gpt, barwidth, label="gpt", color=navyblue)

    ymin = min(ys_baseline + ys_flan + ys_llama + ys_gpt)
    ymax = max(ys_baseline + ys_flan + ys_llama + ys_gpt)
    scale_factor = 1.5
    plt.ylim(max(0, ymin-2*(ymax-ymin)), min(1, ymax+0.5*(ymax-ymin)))

    plt.xticks([2,6,10,14], ["Baseline", "CoT/ZS", "Q2E/FS", "Q2E/ZS"], rotation=0)
    plt.xlabel("Experiment")
    plt.ylabel(score)
    plt.title(name+"\n", fontweight="bold")
    plt.legend(loc="lower right", framealpha=0.95)
#    plt.show()
    plt.savefig("fig/" + name + "-" + score + ".png", dpi=500)
    plt.close()

def barchart_all(data):
    bm25_recs = [data[d]["baselines"]["BM25"]["Recall@1000"] for d in data.keys()]
    bm25_precs = [data[d]["baselines"]["BM25"]["NDCG@10"] for d in data.keys()]
    bm25rm3_recs = [data[d]["baselines"]["BM25+RM3"]["Recall@1000"] for d in data.keys()]
    bm25rm3_precs = [data[d]["baselines"]["BM25+RM3"]["NDCG@10"] for d in data.keys()]
    bm25kl_recs = [data[d]["baselines"]["BM25+KL"]["Recall@1000"] for d in data.keys()]
    bm25kl_precs = [data[d]["baselines"]["BM25+KL"]["NDCG@10"] for d in data.keys()]
    cot_flan_recs = [data[d]["chain-of-thoughts"]["flan-ul2"]["Recall@1000"] for d in data.keys()]
    cot_flan_precs = [data[d]["chain-of-thoughts"]["flan-ul2"]["NDCG@10"] for d in data.keys()]
    cot_llama_recs = [data[d]["chain-of-thoughts"]["llama"]["Recall@1000"] for d in data.keys()]
    cot_llama_precs = [data[d]["chain-of-thoughts"]["llama"]["NDCG@10"] for d in data.keys()]
    cot_gpt_recs = [data[d]["chain-of-thoughts"]["gpt"]["Recall@1000"] for d in data.keys()]
    cot_gpt_precs = [data[d]["chain-of-thoughts"]["gpt"]["NDCG@10"] for d in data.keys()]
    fs_flan_recs = [data[d]["similar-queries-fs"]["flan-ul2"]["Recall@1000"] for d in data.keys()]
    fs_flan_precs = [data[d]["similar-queries-fs"]["flan-ul2"]["NDCG@10"] for d in data.keys()]
    fs_llama_recs = [data[d]["similar-queries-fs"]["llama"]["Recall@1000"] for d in data.keys()]
    fs_llama_precs = [data[d]["similar-queries-fs"]["llama"]["NDCG@10"] for d in data.keys()]
    fs_gpt_recs = [data[d]["similar-queries-fs"]["gpt"]["Recall@1000"] for d in data.keys()]
    fs_gpt_precs = [data[d]["similar-queries-fs"]["gpt"]["NDCG@10"] for d in data.keys()]
    zs_flan_recs = [data[d]["similar-queries-zs"]["flan-ul2"]["Recall@1000"] for d in data.keys()]
    zs_flan_precs = [data[d]["similar-queries-zs"]["flan-ul2"]["NDCG@10"] for d in data.keys()]
    zs_llama_recs = [data[d]["similar-queries-zs"]["llama"]["Recall@1000"] for d in data.keys()]
    zs_llama_precs = [data[d]["similar-queries-zs"]["llama"]["NDCG@10"] for d in data.keys()]
    zs_gpt_recs = [data[d]["similar-queries-zs"]["gpt"]["Recall@1000"] for d in data.keys()]
    zs_gpt_precs = [data[d]["similar-queries-zs"]["gpt"]["NDCG@10"] for d in data.keys()]

    matplotlib.rcParams.update({'font.size': 13})
    plt.figure(figsize=(8, 6))
    barwidth = 0.9
    # recall
    plt.bar(1, mean(bm25_recs), barwidth, yerr=stdev(bm25_recs), capsize=5, color=orange, label="BM25")
    plt.bar(2, mean(bm25rm3_recs), barwidth, yerr=stdev(bm25rm3_recs), capsize=5, color=violet, label="BM25+RM3")
    plt.bar(3, mean(bm25kl_recs), barwidth, yerr=stdev(bm25kl_recs), capsize=5, color=darkorange, label="BM25+KL")
    plt.bar(5, mean(cot_flan_recs), barwidth, yerr=stdev(cot_flan_recs), capsize=5, color=turquoise, label="flan")
    plt.bar(6, mean(cot_llama_recs), barwidth, yerr=stdev(cot_llama_recs), capsize=5, color=purple, label="llama")
    plt.bar(7, mean(cot_gpt_recs), barwidth, yerr=stdev(cot_gpt_recs), capsize=5, color=navyblue, label="gpt")
    plt.bar(9, mean(fs_flan_recs), barwidth, yerr=stdev(fs_flan_recs), capsize=5, color=turquoise)
    plt.bar(10, mean(fs_llama_recs), barwidth, yerr=stdev(fs_llama_recs), capsize=5, color=purple)
    plt.bar(11, mean(fs_gpt_recs), barwidth, yerr=stdev(fs_gpt_recs), capsize=5, color=navyblue)
    plt.bar(13, mean(zs_flan_recs), barwidth, yerr=stdev(zs_flan_recs), capsize=5, color=turquoise)
    plt.bar(14, mean(zs_llama_recs), barwidth, yerr=stdev(zs_llama_recs), capsize=5, color=purple)
    plt.bar(15, mean(zs_gpt_recs), barwidth, yerr=stdev(zs_gpt_recs), capsize=5, color=navyblue)
    plt.hlines([mean(x) for x in [bm25_recs, bm25rm3_recs, bm25kl_recs]], [x-barwidth/2 for x in [1,2,3]], 15+barwidth/2, colors=[orange, violet, darkorange], linestyles="dashed")

    plt.ylim(0.2, 0.9)
    plt.xticks([2,6,10,14], ["Baseline", "CoT/ZS", "Q2E/FS", "Q2E/ZS"], rotation=0)
    plt.xlabel("Experiment")
    plt.ylabel("Recall@1000")
    plt.title("Gemittelte Ergebnisse über alle Datensätze\n", fontweight="bold")
    plt.legend(loc="lower right", framealpha=0.95)
#    plt.show()
    plt.savefig("all-recall.png", dpi=500)
    plt.close()

    #ndcg
    plt.figure(figsize=(8, 6))
    plt.bar(1, mean(bm25_precs), barwidth, yerr=stdev(bm25_precs), capsize=5, color=orange, label="BM25")
    plt.bar(2, mean(bm25rm3_precs), barwidth, yerr=stdev(bm25rm3_precs), capsize=5, color=violet, label="BM25+RM3")
    plt.bar(3, mean(bm25kl_precs), barwidth, yerr=stdev(bm25kl_precs), capsize=5, color=darkorange, label="BM25+KL")
    plt.bar(5, mean(cot_flan_precs), barwidth, yerr=stdev(cot_flan_precs), capsize=5, color=turquoise, label="flan")
    plt.bar(6, mean(cot_llama_precs), barwidth, yerr=stdev(cot_llama_precs), capsize=5, color=purple, label="llama")
    plt.bar(7, mean(cot_gpt_precs), barwidth, yerr=stdev(cot_gpt_precs), capsize=5, color=navyblue, label="gpt")
    plt.bar(9, mean(fs_flan_precs), barwidth, yerr=stdev(fs_flan_precs), capsize=5, color=turquoise)
    plt.bar(10, mean(fs_llama_precs), barwidth, yerr=stdev(fs_llama_precs), capsize=5, color=purple)
    plt.bar(11, mean(fs_gpt_precs), barwidth, yerr=stdev(fs_gpt_precs), capsize=5, color=navyblue)
    plt.bar(13, mean(zs_flan_precs), barwidth, yerr=stdev(zs_flan_precs), capsize=5, color=turquoise)
    plt.bar(14, mean(zs_llama_precs), barwidth, yerr=stdev(zs_llama_precs), capsize=5, color=purple)
    plt.bar(15, mean(zs_gpt_precs), barwidth, yerr=stdev(zs_gpt_precs), capsize=5, color=navyblue)
    plt.hlines([mean(x) for x in [bm25_precs, bm25rm3_precs, bm25kl_precs]], [x-barwidth/2 for x in [1,2,3]], 15+barwidth/2, colors=[orange, violet, darkorange], linestyles="dashed")

#    plt.grid(True, axis="y")
    plt.ylim(0, 0.55)
    plt.xticks([2,6,10,14], ["Baseline", "CoT/ZS", "Q2E/FS", "Q2E/ZS"], rotation=0)
    plt.xlabel("Experiment")
    plt.ylabel("NDCG@10")
    plt.title("Gemittelte Ergebnisse über alle Datensätze\n", fontweight="bold")
    plt.legend(loc="lower right", framealpha=0.95)
#    plt.show()
    plt.savefig("all-ndcg.png", dpi=500)
    plt.close()

def qualplot(data):
    ds_names = data.keys()
    n_abv_recs = []
    n_abv_ndcgs = []
    for d in data:
        best_baseline_rec = max([data[d]["baselines"][m]["Recall@1000"] for m in ["BM25", "BM25+RM3", "BM25+KL"]])
        best_baseline_ndcg = max([data[d]["baselines"][m]["NDCG@10"] for m in ["BM25", "BM25+RM3", "BM25+KL"]])
        n_abv_rec = 0
        n_abv_ndcg = 0
        for exp in list(data[d].keys())[1:]:
            for m in data[d][exp].keys():
                if data[d][exp][m]["Recall@1000"] > best_baseline_rec:
                    n_abv_rec += 1
                if data[d][exp][m]["NDCG@10"] > best_baseline_ndcg:
                    n_abv_ndcg += 1
        n_abv_recs.append(n_abv_rec)
        n_abv_ndcgs.append(n_abv_ndcg)

#    for num, name in sorted(zip(n_abv_recs, ds_names), reverse=True):
#        print(num, name)

    # recall
    quant = sorted(zip(n_abv_recs, ds_names), reverse=True)
    xs = [q[1] for q in quant]
    ys = [q[0] for q in quant]
    matplotlib.rcParams.update({'font.size': 13})
    plt.figure(figsize=(10,6))
    plt.subplots_adjust(left=0.4)
    plt.barh(xs, ys, color=orange)
    plt.xlabel("Anzahl QE > Baseline")
    plt.ylabel("Datensatz")
    plt.title("Qualitative Auswertung für Recall@1000\n", fontweight="bold")
#    plt.show()
    plt.savefig("qualitative-recall.png", dpi=500)
    plt.close()

    # ndcg
    quant = sorted(zip(n_abv_ndcgs, ds_names), reverse=True)
    xs = [q[1] for q in quant]
    ys = [q[0] for q in quant]
    plt.figure(figsize=(10,6))
    plt.subplots_adjust(left=0.4)
    plt.barh(xs, ys, color=darkorange)
    plt.xlabel("Anzahl QE > Baseline")
    plt.ylabel("Datensatz")
    plt.title("Qualitative Auswertung für NDCG@10\n", fontweight="bold")
#    plt.show()
    plt.savefig("qualitative-ndcg.png", dpi=500)
    plt.close()

def plot_methods_qual(data):
    ds_names = data.keys()
    n_abv_recs = {"chain-of-thoughts": {}, "similar-queries-fs": {}, "similar-queries-zs": {}}
    n_abv_ndcgs = {"chain-of-thoughts": {}, "similar-queries-fs": {}, "similar-queries-zs": {}}
    for d in data:
        best_baseline_rec = max([data[d]["baselines"][m]["Recall@1000"] for m in ["BM25", "BM25+RM3", "BM25+KL"]])
        best_baseline_ndcg = max([data[d]["baselines"][m]["NDCG@10"] for m in ["BM25", "BM25+RM3", "BM25+KL"]])
        for exp in list(data[d].keys())[1:]:
            for m in data[d][exp].keys():
                # initialization
                if m not in n_abv_recs[exp].keys():
                    n_abv_recs[exp][m] = 0
                if m not in n_abv_ndcgs[exp].keys():
                    n_abv_ndcgs[exp][m] = 0

                # calculation (rather, counting)
                if data[d][exp][m]["Recall@1000"] > best_baseline_rec:
                    n_abv_recs[exp][m] += 1
                if data[d][exp][m]["NDCG@10"] > best_baseline_ndcg:
                    n_abv_ndcgs[exp][m] += 1

#    print(n_abv_recs)
#    print(n_abv_ndcgs)

    barwidth = 0.9
    matplotlib.rcParams.update({'font.size': 13})
#    plt.bar(6, 10, 2, color=turquoise, alpha=0.3)
    # flan
    xs_flan = [1,5,9]
    plt.bar(xs_flan, [n_abv_recs[e]["flan-ul2"] for e in n_abv_recs.keys()], barwidth, color=orange, label="flan")
    # llama
    xs_llama = [2,6,10]
    plt.bar(xs_llama, [n_abv_recs[e]["llama"] for e in n_abv_recs.keys()], barwidth, color=violet, label="llama")
    # gpt
    xs_gpt = [3,7,11]
    plt.bar(xs_gpt, [n_abv_recs[e]["gpt"] for e in n_abv_recs.keys()], barwidth, color=darkorange, label="gpt")
    plt.legend(loc="lower right")
    plt.title("Qualitative Recall@1000 Performance nach\nMethode/Model", fontweight="bold")
    plt.xticks([2,6,10], ["CoT/ZS", "Q2E/FS", "Q2E/ZS"], rotation=0)
    plt.xlabel("Experiment")
    plt.ylabel("Anzahl QE > Baseline")
#    plt.show()
    plt.savefig("miregal.png", dpi=500)
    plt.close()

def main():
    data = load_evals()
    datasets = list(data.keys())
    experiments = list(data[datasets[0]].keys())
    scores = ["Recall@1000", "NDCG@10"]

#    for d in datasets:
#        for s in scores:
#            barchart(data, d, s)

#    barchart_all(data)
#    qualplot(data)
    plot_methods_qual(data)

if __name__ == "__main__":
    main()
