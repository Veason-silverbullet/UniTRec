import torch
from scipy.stats import rankdata
from torch.nn.functional import log_softmax, softmax


class AvgMetric:
    def __init__(self, auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10):
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10
        self.hr1 = hr1
        self.hr5 = hr5
        self.hr10 = hr10
        self.avg = (self.auc + self.mrr + (self.ndcg5 + self.ndcg10) / 2 + (self.hr1 * 2 + self.hr5 + self.hr10) / 4) / 4

    def __gt__(self, value):
        return self.avg > value.avg

    def __ge__(self, value):
        return self.avg >= value.avg

    def __lt__(self, value):
        return self.avg < value.avg

    def __le__(self, value):
        return self.avg <= value.avg

    def __str__(self):
        return '%.4f\nAUC = %.4f\nMRR = %.4f\nnDCG@5 = %.4f\nnDCG@10 = %.4f\nHR@1 = %.4f\nHR@5 = %.4f\nHR@10 = %.4f' % (self.avg, self.auc, self.mrr, self.ndcg5, self.ndcg10, self.hr1, self.hr5, self.hr10)


def ordinal_average_ranking(ppl_score: list, dis_score: list) -> list:
    ppl_rank = rankdata(ppl_score)
    dis_rank = rankdata(dis_score)
    average_rank = ppl_rank + dis_rank
    average_rank = rankdata(-(average_rank * (len(ppl_score) + 1) + ppl_rank), method='ordinal').tolist() # a little trick to discern equal ordinal ranking
    return average_rank


# geometric average: https://en.wikipedia.org/wiki/Geometric_mean
def normalized_average_ranking(ppl_score: list, dis_score: list) -> list:
    normalized_log_ppl_score = log_softmax(torch.FloatTensor(ppl_score), dim=0)
    normalized_log_dis_score = log_softmax(torch.FloatTensor(dis_score), dim=0)
    normalized_log_score = normalized_log_ppl_score + normalized_log_dis_score
    average_rank = rankdata((-normalized_log_score).tolist(), method='ordinal').tolist()
    return average_rank


# harmonic average: https://en.wikipedia.org/wiki/Harmonic_mean
def harmonic_average_ranking(ppl_score: list, dis_score: list) -> list:
    normalized_ppl_score = softmax(torch.FloatTensor(ppl_score), dim=0)
    normalized_dis_score = softmax(torch.FloatTensor(dis_score), dim=0)
    harmonic_score = 1 / normalized_ppl_score + 1 / normalized_dis_score
    average_rank = rankdata(harmonic_score.tolist(), method='ordinal').tolist()
    return average_rank


class AverageRanking:
    average_ranking_method = None
    def set_average_ranking_method(average_ranking_method):
        assert average_ranking_method in ['ordinal', 'normalized', 'harmonic']
        if average_ranking_method == 'ordinal':
            AverageRanking.average_ranking_method = ordinal_average_ranking
        elif average_ranking_method == 'normalized':
            AverageRanking.average_ranking_method = normalized_average_ranking
        elif average_ranking_method == 'harmonic':
            AverageRanking.average_ranking_method = harmonic_average_ranking

    def rank(ppl_score: list, dis_score: list):
        return AverageRanking.average_ranking_method(ppl_score, dis_score)


def write_predictions(result_file: str, rankings: list):
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, ranking in enumerate(rankings):
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(ranking).replace(' ', ''))
