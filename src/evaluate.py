import json
import argparse
from math import log2


def mrr(ranks, k):
    rec_ranks = [1./r if r <= k else 0. for r in ranks]
    return sum(rec_ranks) / len(ranks)

def recall(ranks, k):
    return sum(r <= k for r in ranks) / len(ranks)

def ndcg(ranks, k):
    ndcg_per_query = sum(1 / log2(r + 1) for r in ranks if r <= k)
    return ndcg_per_query / len(ranks)


def main(args):
    ranks = []
    for f in args.input_files:
        preds = json.load(open(f))
        for cid in preds:
            sorted_preds = [  # paper ids sorted by recommendation score
                i[0] for i in sorted(preds[cid], key=lambda x: x[1], reverse=True)
            ]
            rank = sorted_preds.index(cid.split('_')[1]) + 1  # rank of correct recommendation
            ranks.append(rank)

    print(f'Recall@k: {recall(ranks, k=10):.5f}')
    print(f'MRR: {mrr(ranks, k=10):.5f}')
    print(f'NDCG: {ndcg(ranks, k=10):.5f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs="+", help='JSON files containing recommendation scores')
    args = parser.parse_args()

    main(args)