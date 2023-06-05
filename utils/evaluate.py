import numpy as np
import json
from sklearn.metrics import roc_auc_score


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


# following paper https://aclanthology.org/2021.acl-short.95.pdf and https://aclanthology.org/2022.acl-long.27.pdf to calcuate nDCG for one-hot recommended item
def ndcg_score_onehot(y_label, y_rank, k=10):
    r = y_rank[y_label]
    if r > k:
        return 0
    else:
        return 1 / np.log2(r + 1)


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def hit_ratio_score(y_true, y_rank, k=10):
    hit_num = 0
    for i in range(len(y_rank)):
        if bool(y_true[i]) and y_rank[i] <= k:
            hit_num += 1
    return hit_num / np.sum(y_true)


def parse_line(l):
    impid, ranks = l.strip('\n').split()
    ranks = json.loads(ranks)
    return impid, ranks


def scoring(truth_f, sub_f, onehot=False):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    hr1s = []
    hr5s = []
    hr10s = []

    for line_index, lt in enumerate(truth_f):
        ls = sub_f.readline()
        impid, labels = parse_line(lt)
        # ignore masked impressions
        if labels == []:
            continue
        if ls == '':
            # empty line: filled with 0 ranks
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except:
                raise ValueError("line-{}: Invalid Input Format!".format(line_index + 1))
        if sub_impid != impid:
            raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(line_index + 1, sub_impid, impid))

        lt_len = float(len(labels))
        y_true = np.array(labels, dtype='float32')
        y_score = []
        for rank in sub_ranks:
            score_rslt = 1.0 / rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError("Line-{}: score_rslt should be int from 0 to {}".format(line_index + 1, lt_len))
            y_score.append(score_rslt)

        auc = roc_auc_score(y_true, y_score)
        mrr = mrr_score(y_true, y_score)
        if not onehot:
            ndcg5 = ndcg_score(y_true, y_score, 5)
            ndcg10 = ndcg_score(y_true, y_score, 10)
        else:
            onehots = []
            for i in range(len(y_true)):
                if y_true[i] == 1:
                    onehots.append(i)
            assert len(onehots) == 1
            y_label = onehots[0]
            ndcg5 = ndcg_score_onehot(y_label, sub_ranks, 5)
            ndcg10 = ndcg_score_onehot(y_label, sub_ranks, 10)
        hr1 = hit_ratio_score(y_true, sub_ranks, 1)
        hr5 = hit_ratio_score(y_true, sub_ranks, 5)
        hr10 = hit_ratio_score(y_true, sub_ranks, 10)
        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)
        hr1s.append(hr1)
        hr5s.append(hr5)
        hr10s.append(hr10)
    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s), np.mean(hr1s), np.mean(hr5s), np.mean(hr10s)
