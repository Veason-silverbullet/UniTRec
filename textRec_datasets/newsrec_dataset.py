import os
import pickle
import numpy as np
from copy import deepcopy
import random
import torch.utils.data as data


class NewsRecTrainDataset(data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.PAD_TOKEN_ID = 1
        self.IGNORE_TOKEN_ID = -100
        with open(os.path.join('textRec_datasets', args.task, 'history_input_ids-train-%d.pkl' % args.encoder_seq_len), 'rb') as f:
            self.history_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_segment_ids-train-%d.pkl' % args.encoder_seq_len), 'rb') as f:
            self.history_segment_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_global_attention_mask-train-%d.pkl' % args.encoder_seq_len), 'rb') as f:
            self.history_global_attention_mask = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_local_position_ids-train-%d.pkl' % args.encoder_seq_len), 'rb') as f:
            self.history_local_position_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'news_input_ids-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.news_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'news_cls_indices-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.news_cls_indices = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'targets-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.targets = pickle.load(f)
            self.targets[self.targets == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID
        with open(os.path.join('textRec_datasets', args.task, 'indexing-train.pkl'), 'rb') as f:
            self.indexing = pickle.load(f)
        self.negative_sample_num = args.negative_sample_num
        for i in range(len(self.indexing)):
            required_negative_sample_num = self.negative_sample_num * len(self.indexing[i][1]) * args.epoch
            if len(self.indexing[i][2]) < required_negative_sample_num:
                negative_samples = deepcopy(self.indexing[i][2])
                self.indexing[i][2] = []
                while len(self.indexing[i][2]) < required_negative_sample_num:
                    random.shuffle(negative_samples)
                    self.indexing[i][2].extend(negative_samples)

    def negative_sampling(self, epoch):
        self.samples = []
        for hisotory_index, positive_samples, negative_samples in self.indexing:
            N = len(positive_samples)
            for i, positive_sample in enumerate(positive_samples):
                start_index = self.negative_sample_num * ((epoch - 1) * N + i)
                end_index = start_index + self.negative_sample_num
                self.samples.append([hisotory_index, [positive_sample] + negative_samples[start_index: end_index]])
        self.num = len(self.samples)

    def __len__(self):
        return self.num

    # history_input_ids             : [encoder_seq_len]
    # history_segment_ids           : [encoder_seq_len]
    # history_global_attention_mask : [encoder_seq_len]
    # history_local_position_ids    : [encoder_seq_len]
    # news_input_ids                : [1 + negative_sample_num, decoder_seq_len]
    # news_cls_indices              : [1 + negative_sample_num]
    # targets                       : [1 + negative_sample_num, decoder_seq_len]
    def __getitem__(self, index):
        history_index, sample_indices = self.samples[index]
        return self.history_input_ids[history_index].astype(np.int32), self.history_segment_ids[history_index], self.history_global_attention_mask[history_index], self.history_local_position_ids[history_index].astype(np.int32), \
               self.news_input_ids[sample_indices], self.news_cls_indices[sample_indices], self.targets[sample_indices]


class NewsRecValDataset(data.Dataset):
    def __init__(self, args, mode, rank=None, world_size=1):
        super().__init__()
        self.PAD_TOKEN_ID = 1
        self.IGNORE_TOKEN_ID = -100
        self.mode = mode
        self.rank = rank
        self.world_size = world_size
        assert self.mode in ['dev', 'test']
        assert self.world_size == 1 or (self.rank is not None and self.rank < self.world_size)
        self.truth_file = os.path.join('textRec_datasets', args.task, self.mode, 'truth.txt')
        with open(os.path.join('textRec_datasets', args.task, 'history_input_ids-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_segment_ids-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_segment_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_global_attention_mask-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_global_attention_mask = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_local_position_ids-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_local_position_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'news_input_ids-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.news_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'news_cls_indices-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.news_cls_indices = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'targets-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.targets = pickle.load(f)
            self.targets[self.targets == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID
        with open(os.path.join('textRec_datasets', args.task, 'indexing-%s.pkl' % self.mode), 'rb') as f:
            self.indexing = pickle.load(f)
        if self.world_size > 1:
            num = len(self.indexing)
            start_index = num // self.world_size * self.rank
            end_index = (num // self.world_size * (self.rank + 1)) if self.rank != self.world_size - 1 else num
            self.indexing = self.indexing[start_index: end_index]
        self.indices = []
        for i, (history_index, news_indices) in enumerate(self.indexing):
            for _ in range(len(news_indices)):
                self.indices.append(i)
        self.num = len(self.indexing)

    def __len__(self):
        return self.num

    # history_input_ids             : [encoder_seq_len]
    # history_segment_ids           : [encoder_seq_len]
    # history_global_attention_mask : [encoder_seq_len]
    # history_local_position_ids    : [encoder_seq_len]
    # news_input_ids                : [news_num, decoder_seq_len]
    # news_cls_indices              : [news_num]
    # targets                       : [news_num, decoder_seq_len]
    # assert batch_size == 1 and news_num <= 300
    def __getitem__(self, index):
        history_index, news_indices = self.indexing[index]
        return self.history_input_ids[history_index].astype(np.int32), self.history_segment_ids[history_index], self.history_global_attention_mask[history_index], self.history_local_position_ids[history_index].astype(np.int32), \
               self.news_input_ids[news_indices], self.news_cls_indices[news_indices], self.targets[news_indices]
