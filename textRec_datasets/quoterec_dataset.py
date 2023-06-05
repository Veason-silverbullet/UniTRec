import os
import pickle
import json
import numpy as np
import random
import torch.utils.data as data


class QuoteRecTrainDataset(data.Dataset):
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
        with open(os.path.join('textRec_datasets', args.task, 'quote_input_ids-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.quote_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'quote_cls_indices-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.quote_cls_indices = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'targets-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.targets = pickle.load(f)
            self.targets[self.targets == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID
        with open(os.path.join('textRec_datasets', args.task, 'info.json'), 'r', encoding='utf-8') as f:
            info = json.load(f)
            self.candidate_num = len(info['quote_map'])
            self.quote_indices = info['train_quote_indices']
            self.num = len(self.quote_indices)
        self.negative_sample_num = args.negative_sample_num
        required_negative_sample_num = self.negative_sample_num * args.epoch
        self.negative_samples = []
        for i in range(self.num):
            self.negative_samples.append([])
            while True:
                negative_samples = []
                for j in range(self.candidate_num):
                    if j != self.quote_indices[i]:
                        negative_samples.append(j)
                random.shuffle(negative_samples)
                self.negative_samples[i].extend(negative_samples)
                if len(self.negative_samples[i]) >= required_negative_sample_num:
                    break

    def negative_sampling(self, epoch):
        self.samples = []
        for i, quote_index in enumerate(self.quote_indices):
            self.samples.append([quote_index] + self.negative_samples[i][(epoch - 1) * self.negative_sample_num: epoch * self.negative_sample_num])

    def __len__(self):
        return self.num

    # history_input_ids             : [encoder_seq_len]
    # history_segment_ids           : [encoder_seq_len]
    # history_global_attention_mask : [encoder_seq_len]
    # history_local_position_ids    : [encoder_seq_len]
    # quote_input_ids               : [1 + negative_sample_num, decoder_seq_len]
    # quote_cls_indices             : [1 + negative_sample_num]
    # targets                       : [1 + negative_sample_num, decoder_seq_len]
    def __getitem__(self, index):
        sample_indices = self.samples[index]
        return self.history_input_ids[index].astype(np.int32), self.history_segment_ids[index], self.history_global_attention_mask[index], self.history_local_position_ids[index].astype(np.int32), \
               self.quote_input_ids[sample_indices], self.quote_cls_indices[sample_indices], self.targets[sample_indices]


class QuoteRecValDataset(data.Dataset):
    def __init__(self, args, mode, rank=None, world_size=1):
        super().__init__()
        self.PAD_TOKEN_ID = 1
        self.IGNORE_TOKEN_ID = -100
        self.mode = mode
        self.rank = rank
        self.world_size = world_size
        assert self.mode in ['dev', 'test']
        assert self.world_size == 1 or (self.rank is not None and self.rank < self.world_size)
        self.VAL_QUOTE_SEGMENT_NUM = 400
        self.truth_file = os.path.join('textRec_datasets', args.task, 'truth-' + mode + '.txt')
        with open(os.path.join('textRec_datasets', args.task, 'history_input_ids-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_segment_ids-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_segment_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_global_attention_mask-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_global_attention_mask = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'history_local_position_ids-%s-%d.pkl' % (self.mode, args.encoder_seq_len)), 'rb') as f:
            self.history_local_position_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'quote_input_ids-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.quote_input_ids = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'quote_cls_indices-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.quote_cls_indices = pickle.load(f)
        with open(os.path.join('textRec_datasets', args.task, 'targets-%d.pkl' % args.decoder_seq_len), 'rb') as f:
            self.targets = pickle.load(f)
            self.targets[self.targets == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID
        with open(os.path.join('textRec_datasets', args.task, 'info.json'), 'r', encoding='utf-8') as f:
            info = json.load(f)
            self.candidate_num = len(info['quote_map'])
            self.quote_indices = info[self.mode + '_quote_indices']
        if self.world_size > 1:
            num = len(self.quote_indices)
            start_index = num // self.world_size * self.rank
            end_index = (num // self.world_size * (self.rank + 1)) if self.rank != self.world_size - 1 else num
            self.history_input_ids = self.history_input_ids[start_index: end_index]
            self.history_segment_ids = self.history_segment_ids[start_index: end_index]
            self.history_global_attention_mask = self.history_global_attention_mask[start_index: end_index]
            self.history_local_position_ids = self.history_local_position_ids[start_index: end_index]
            self.quote_indices = self.quote_indices[start_index: end_index]
        self.indexing = []
        self.indices = []
        for i in range(len(self.quote_indices)):
            start_index = 0
            while start_index < self.candidate_num:
                end_index = min(start_index + self.VAL_QUOTE_SEGMENT_NUM, self.candidate_num)
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
    # quote_input_ids               : [VAL_QUOTE_SEGMENT_NUM, decoder_seq_len]
    # quote_cls_indices             : [VAL_QUOTE_SEGMENT_NUM]
    # targets                       : [VAL_QUOTE_SEGMENT_NUM, decoder_seq_len]
    # assert batch_size == 1 and VAL_QUOTE_SEGMENT_NUM <= 400
    def __getitem__(self, index):
        history_index, quote_indices = self.indexing[index]
        return self.history_input_ids[history_index].astype(np.int32), self.history_segment_ids[history_index], self.history_global_attention_mask[history_index], self.history_local_position_ids[history_index].astype(np.int32), \
               self.quote_input_ids[quote_indices], self.quote_cls_indices[quote_indices], self.targets[quote_indices]
