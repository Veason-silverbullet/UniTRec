import os
import pickle
import json
import numpy as np
import random
from copy import deepcopy
import torch.utils.data as data


class EngageRecTrainDataset(data.Dataset):
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
        with open(os.path.join('textRec_datasets', args.task, 'engage_input_ids-train-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.engage_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'engage_cls_indices-train-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.engage_cls_indices = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'targets-train-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.targets = pickle.load(f)
            self.targets[self.targets == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID
        with open(os.path.join('textRec_datasets', args.task, 'history_user_map-train.json'), 'r') as f:
            history_user_map = json.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'train_meta.pkl'), 'rb') as f:
            meta_data = pickle.load(f)
        self.negative_sample_num = args.negative_sample_num
        required_negative_sample_num = self.negative_sample_num * args.epoch
        self.candidate_num = self.engage_input_ids.shape[0]
        assert self.candidate_num == self.engage_cls_indices.shape[0] and self.candidate_num == self.targets.shape[0]
        self.history_indices = []
        self.positive_samples = []
        self.negative_samples = []
        for index, user_id in enumerate(meta_data):
            assert history_user_map[str(user_id)] == index
            history_turns, candidates = meta_data[user_id]
            negative_samples = []
            for i in range(self.candidate_num):
                if i not in candidates:
                    negative_samples.append(i)
            N = required_negative_sample_num * len(candidates)
            if len(negative_samples) >= N:
                random.shuffle(negative_samples)
            else:
                negative_samples_ = []
                while N > 0:
                    samples = deepcopy(negative_samples)
                    random.shuffle(samples)
                    negative_samples_.extend(samples)
                    N -= len(samples)
                negative_samples = negative_samples_
            for i, candidate in enumerate(candidates):
                self.history_indices.append(index)
                self.positive_samples.append(candidate)
                self.negative_samples.append(negative_samples[i * required_negative_sample_num: (i + 1) * required_negative_sample_num])
        self.num = len(self.history_indices)

    def negative_sampling(self, epoch):
        self.samples = []
        for i in range(self.num):
            self.samples.append([self.positive_samples[i]] + self.negative_samples[i][(epoch - 1) * self.negative_sample_num: epoch * self.negative_sample_num])

    def __len__(self):
        return self.num

    # history_input_ids             : [encoder_seq_len]
    # history_segment_ids           : [encoder_seq_len]
    # history_global_attention_mask : [encoder_seq_len]
    # history_local_position_ids    : [encoder_seq_len]
    # engage_input_ids              : [1 + negative_sample_num, decoder_seq_len]
    # engage_cls_indices            : [1 + negative_sample_num]
    # targets                       : [1 + negative_sample_num, decoder_seq_len]
    def __getitem__(self, index):
        history_index = self.history_indices[index]
        sample_indices = self.samples[index]
        return self.history_input_ids[history_index].astype(np.int32), self.history_segment_ids[history_index], self.history_global_attention_mask[history_index], self.history_local_position_ids[history_index].astype(np.int32), \
               self.engage_input_ids[sample_indices], self.engage_cls_indices[sample_indices], self.targets[sample_indices]


class EngageRecValDataset(data.Dataset):
    def __init__(self, args, mode, rank=None, world_size=1):
        super().__init__()
        self.PAD_TOKEN_ID = 1
        self.IGNORE_TOKEN_ID = -100
        self.mode = mode
        self.rank = rank
        self.world_size = world_size
        assert self.mode in ['dev', 'test']
        assert self.world_size == 1 or (self.rank is not None and self.rank < self.world_size)
        self.VAL_ENGAGE_SEGMENT_NUM = 64
        self.truth_file = os.path.join('textRec_datasets', args.task, 'truth-' + self.mode + '.txt')
        with open(os.path.join('textRec_datasets', args.task, 'history_input_ids-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_segment_ids-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_segment_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_global_attention_mask-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_global_attention_mask = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_local_position_ids-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_local_position_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'engage_input_ids-%s-%d.pkl' % (self.mode, args.decoder_seq_len)), 'rb') as f:
            self.engage_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'engage_cls_indices-%s-%d.pkl' % (self.mode, args.decoder_seq_len)), 'rb') as f:
            self.engage_cls_indices = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'targets-%s-%d.pkl' % (self.mode, args.decoder_seq_len)), 'rb') as f:
            self.targets = pickle.load(f)
            self.targets[self.targets == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID
        if self.world_size > 1:
            num = self.history_input_ids.shape[0]
            assert num == self.history_segment_ids.shape[0] and num == self.history_global_attention_mask.shape[0] and num == self.history_local_position_ids.shape[0]
            start_index = num // self.world_size * self.rank
            end_index = (num // self.world_size * (self.rank + 1)) if self.rank != self.world_size - 1 else num
            self.history_input_ids = self.history_input_ids[start_index: end_index]
            self.history_segment_ids = self.history_segment_ids[start_index: end_index]
            self.history_global_attention_mask = self.history_global_attention_mask[start_index: end_index]
            self.history_local_position_ids = self.history_local_position_ids[start_index: end_index]
        self.candidate_num = self.engage_input_ids.shape[0]
        assert self.candidate_num == self.engage_cls_indices.shape[0] and self.candidate_num == self.targets.shape[0]
        self.indexing = []
        self.indices = []
        for i in range(self.history_input_ids.shape[0]):
            start_index = 0
            while start_index < self.candidate_num:
                end_index = min(start_index + self.VAL_ENGAGE_SEGMENT_NUM, self.candidate_num)
                self.indexing.append([i, [j for j in range(start_index, end_index)]])
                if end_index == self.candidate_num:
                    break
                start_index = end_index
            self.indices.extend([i for j in range(self.candidate_num)])
        self.num = len(self.indexing)

    def __len__(self):
        return self.num

    # history_input_ids             : [encoder_seq_len]
    # history_segment_ids           : [encoder_seq_len]
    # history_global_attention_mask : [encoder_seq_len]
    # history_local_position_ids    : [encoder_seq_len]
    # engage_input_ids              : [VAL_ENGAGE_SEGMENT_NUM, decoder_seq_len]
    # engage_cls_indices            : [VAL_ENGAGE_SEGMENT_NUM]
    # targets                       : [VAL_ENGAGE_SEGMENT_NUM, decoder_seq_len]
    # assert batch_size == 1 and VAL_ENGAGE_SEGMENT_NUM <= 64
    def __getitem__(self, index):
        history_index, engage_indices = self.indexing[index]
        return self.history_input_ids[history_index].astype(np.int32), self.history_segment_ids[history_index], self.history_global_attention_mask[history_index], self.history_local_position_ids[history_index].astype(np.int32), \
               self.engage_input_ids[engage_indices], self.engage_cls_indices[engage_indices], self.targets[engage_indices]
