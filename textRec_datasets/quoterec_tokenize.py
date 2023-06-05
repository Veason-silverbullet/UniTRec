import sys
sys.path.append('..')
import os
import json
import pickle
import numpy as np
from transformers.models.bart import BartTokenizer
BOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
DECODER_START_TOKEN_ID = 2 # a unique hack of BART, also refer to https://github.com/huggingface/transformers/issues/5212
IGNORE_TOKEN_ID = -100
MAX_HISTORY_NUM = 255
MAX_IMPRESSION_NUM = 300
quoterec_root = 'quoteRec'
if not os.path.exists(quoterec_root):
    os.mkdir(quoterec_root)
reddit_quote_dataset_root = os.path.join(quoterec_root, 'Reddit-quote')
quoter_dataset_root = os.path.join(quoterec_root, 'QuoteR')
if not os.path.exists(reddit_quote_dataset_root):
    os.mkdir(reddit_quote_dataset_root)
if not os.path.exists(quoter_dataset_root):
    os.mkdir(quoter_dataset_root)
if not os.path.exists('../../download'):
    os.mkdir('../../download')
if not os.path.exists('../../backbone_models'):
    os.mkdir('../../backbone_models')
if not os.path.exists('../../backbone_models/bart-base'):
    os.system('git lfs install')
    os.system('git clone https://huggingface.co/facebook/bart-base ../../backbone_models/bart-base')
tokenizer = BartTokenizer.from_pretrained('../../backbone_models/bart-base')


def download_QuoteDatasets():
    reddit_quote_data_file = '../../download/Reddit_data.json'
    if not os.path.exists(reddit_quote_data_file):
        os.system('wget https://raw.githubusercontent.com/Lingzhi-WANG/Datasets-for-Quotation-Recommendation/main/Reddit_data.json -P ../../download')
    quoter_data_file = '../../download/english.txt'
    if not os.path.exists(quoter_data_file):
        os.system('wget https://github.com/thunlp/QuoteR/raw/main/data/english.txt -P ../../download')
    reddit_quote_train_file = os.path.join(reddit_quote_dataset_root, 'train.json')
    reddit_quote_dev_file = os.path.join(reddit_quote_dataset_root, 'dev.json')
    reddit_quote_test_file = os.path.join(reddit_quote_dataset_root, 'test.json')
    reddit_quote_info_file = os.path.join(reddit_quote_dataset_root, 'info.json')
    if not all(map(os.path.exists, [reddit_quote_train_file, reddit_quote_dev_file, reddit_quote_test_file, reddit_quote_info_file])):
        with open(reddit_quote_data_file, 'r', encoding='utf-8') as reddit_quote_f, \
            open(reddit_quote_train_file, 'w', encoding='utf-8') as reddit_quote_train_f, \
            open(reddit_quote_dev_file, 'w', encoding='utf-8') as reddit_quote_dev_f, \
            open(reddit_quote_test_file, 'w', encoding='utf-8') as reddit_quote_test_f, \
            open(reddit_quote_info_file, 'w', encoding='utf-8') as reddit_quote_info_f:
            quote_map = {}
            train_quote_indices = []
            dev_quote_indices = []
            test_quote_indices = []
            for i, line in enumerate(reddit_quote_f):
                item = line.strip()
                if len(item) > 0:
                    history, quote, info = json.loads(item)
                    assert type(history) == list and type(quote) == str and len(info) == 2 and info[0] == 'null' and info[1] == 'null', item
                    quote = quote.strip()
                    if quote not in quote_map:
                        quote_map[quote] = len(quote_map)
                    history = list(filter(lambda x: x != '', history))
                    assert len(history) > 0
                    if i % 10 < 8: # 80% for train
                        reddit_quote_train_f.write(json.dumps([history, quote_map[quote]]) + '\n')
                        train_quote_indices.append(quote_map[quote])
                    elif i % 10 == 8: # 10% for dev
                        reddit_quote_dev_f.write(json.dumps([history, quote_map[quote]]) + '\n')
                        dev_quote_indices.append(quote_map[quote])
                    else: # 10% for test
                        reddit_quote_test_f.write(json.dumps([history, quote_map[quote]]) + '\n')
                        test_quote_indices.append(quote_map[quote])
            assert len(quote_map) == 1111, 'check if the quote number matches paper https://aclanthology.org/2021.acl-short.95.pdf'
            json.dump({
                'quote_map': quote_map,
                'train_quote_indices': train_quote_indices,
                'dev_quote_indices': dev_quote_indices,
                'test_quote_indices': test_quote_indices
            }, reddit_quote_info_f)
    quoter_train_file = os.path.join(quoter_dataset_root, 'train.json')
    quoter_dev_file = os.path.join(quoter_dataset_root, 'dev.json')
    quoter_test_file = os.path.join(quoter_dataset_root, 'test.json')
    quoter_info_file = os.path.join(quoter_dataset_root, 'info.json')
    if not all(map(os.path.exists, [quoter_train_file, quoter_dev_file, quoter_test_file, quoter_info_file])):
        with open(quoter_data_file, 'r', encoding='utf-8') as quoter_f, \
            open(quoter_train_file, 'w', encoding='utf-8') as quoter_train_f, \
            open(quoter_dev_file, 'w', encoding='utf-8') as quoter_dev_f, \
            open(quoter_test_file, 'w', encoding='utf-8') as quoter_test_f, \
            open(quoter_info_file, 'w', encoding='utf-8') as quoter_info_f:
            quote_map = {}
            train_quote_indices = []
            dev_quote_indices = []
            test_quote_indices = []
            for i, line in enumerate(quoter_f):
                item = line.strip()
                if len(item) > 0:
                    items = line.strip('\n').split('\t')
                    assert len(items) == 3
                    left_history, quote, right_history = items[0].strip(), items[1].strip().lower(), items[2].strip()
                    assert left_history != '' and quote != '' and right_history != ''
                    history = [left_history, right_history]
                    quote = quote.strip()
                    if quote not in quote_map:
                        quote_map[quote] = len(quote_map)
                    if i < 101171:
                        quoter_train_f.write(json.dumps([history, quote_map[quote]]) + '\n')
                        train_quote_indices.append(quote_map[quote])
                    elif 101171 <= i < 113942:
                        quoter_dev_f.write(json.dumps([history, quote_map[quote]]) + '\n')
                        dev_quote_indices.append(quote_map[quote])
                    else:
                        quoter_test_f.write(json.dumps([history, quote_map[quote]]) + '\n')
                        test_quote_indices.append(quote_map[quote])
            assert len(quote_map) == 6108, 'check if the quote number matches paper https://aclanthology.org/2022.acl-long.27.pdf'
            json.dump({
                'quote_map': quote_map,
                'train_quote_indices': train_quote_indices,
                'dev_quote_indices': dev_quote_indices,
                'test_quote_indices': test_quote_indices
            }, quoter_info_f)


def preprocess_datasets(encoder_max_lengths, decoder_max_lengths):
    for dataset_root in [reddit_quote_dataset_root, quoter_dataset_root]:
        # 1. quote token preprocessing
        encoder_max_length = encoder_max_lengths[dataset_root]
        decoder_max_length = decoder_max_lengths[dataset_root]
        quote_input_ids_file = os.path.join(dataset_root, 'quote_input_ids-%d.pkl' % decoder_max_length)
        quote_cls_indices_file = os.path.join(dataset_root, 'quote_cls_indices-%d.pkl' % decoder_max_length)
        quote_attention_mask_file = os.path.join(dataset_root, 'quote_attention_mask-%d.pkl' % decoder_max_length)
        targets_file = os.path.join(dataset_root, 'targets-%d.pkl' % decoder_max_length)
        with open(os.path.join(dataset_root, 'info.json'), 'r', encoding='utf-8') as f:
            info = json.load(f)
            quote_map = info['quote_map']
            num = len(quote_map)
        quote_input_ids = np.full([num, decoder_max_length], PAD_TOKEN_ID, dtype=np.int32)
        quote_cls_indices = np.zeros([num], dtype=np.int32)
        quote_attention_mask = np.zeros([num, decoder_max_length], dtype=bool)
        targets = np.full([num, decoder_max_length], PAD_TOKEN_ID, dtype=np.int64)
        quote_input_ids[:, 0] = DECODER_START_TOKEN_ID
        target_token_num = 0
        for quote in quote_map:
            while '  ' in quote:
                quote = quote.replace('  ', ' ')
            input_ids = tokenizer(quote, add_special_tokens=False)['input_ids'] + [EOS_TOKEN_ID]
            index = quote_map[quote]
            token_num = len(input_ids)
            assert token_num + 1 <= decoder_max_length, token_num + 1
            for i in range(token_num):
                quote_input_ids[index][i + 1] = input_ids[i]
                targets[index][i] = input_ids[i]
                target_token_num += 1
            quote_cls_indices[index] = token_num
            quote_attention_mask[index][:token_num + 1] = True
            assert targets[index][token_num - 1] == EOS_TOKEN_ID
            targets[index][token_num - 1] = IGNORE_TOKEN_ID
        with open(quote_input_ids_file, 'wb') as f:
            pickle.dump(quote_input_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(quote_cls_indices_file, 'wb') as f:
            pickle.dump(quote_cls_indices, f)
        with open(quote_attention_mask_file, 'wb') as f:
            pickle.dump(quote_attention_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(targets_file, 'wb') as f:
            pickle.dump(targets, f, protocol=pickle.HIGHEST_PROTOCOL)
        assert ((targets != PAD_TOKEN_ID) & (targets != IGNORE_TOKEN_ID)).any(axis=1).all()
        print(dataset_root, ': target_token_num =', target_token_num / num)
        # 2. history token preprocessing
        max_history_length = 0
        for mode in ['train', 'dev', 'test']:
            history_input_ids_file = os.path.join(dataset_root, 'history_input_ids-%s-%d.pkl' % (mode, encoder_max_length))
            history_segment_ids_file = os.path.join(dataset_root, 'history_segment_ids-%s-%d.pkl' % (mode, encoder_max_length))
            history_global_attention_mask_file = os.path.join(dataset_root, 'history_global_attention_mask-%s-%d.pkl' % (mode, encoder_max_length))
            history_local_position_ids_file = os.path.join(dataset_root, 'history_local_position_ids-%s-%d.pkl' % (mode, encoder_max_length))
            num = 0
            with open(os.path.join(dataset_root, mode + '.json'), 'r', encoding='utf-8') as f:
                for line in f:
                    item = line.strip()
                    if len(item) > 0:
                        num += 1
            print('Preprocessing %s %s : %d' % (dataset_root, mode, num))
            history_input_ids = np.full([num, encoder_max_length], PAD_TOKEN_ID, dtype=np.uint16)
            history_segment_ids = np.full([num, encoder_max_length], MAX_HISTORY_NUM, dtype=np.uint8)
            history_global_attention_mask = np.zeros([num, encoder_max_length], dtype=bool)
            history_local_position_ids = np.zeros([num, encoder_max_length], dtype=np.int16)
            history_input_ids[:, 0] = BOS_TOKEN_ID
            history_segment_ids[:, 0] = 0
            history_global_attention_mask[:, 0] = True
            with open(os.path.join(dataset_root, mode + '.json'), 'r', encoding='utf-8') as f:
                for index, line in enumerate(f):
                    item = line.strip()
                    if len(item) > 0:
                        history, quote_index = json.loads(item)
                        pos = 1
                        for history_index, history_document in enumerate(history):
                            while '  ' in history_document:
                                history_document = history_document.replace('  ', ' ')
                            history_tokens = tokenizer(history_document, add_special_tokens=False)['input_ids'] + [EOS_TOKEN_ID]
                            history_token_num = len(history_tokens)
                            if pos + history_token_num < encoder_max_length:
                                for i in range(history_token_num):
                                    history_input_ids[index][pos] = history_tokens[i]
                                    history_segment_ids[index][pos] = history_index + 1
                                    history_global_attention_mask[index][pos] = True
                                    history_local_position_ids[index][pos] = i
                                    pos += 1
                            else:
                                for i in range(encoder_max_length - pos):
                                    history_input_ids[index][pos] = history_tokens[i]
                                    history_segment_ids[index][pos] = history_index + 1
                                    history_global_attention_mask[index][pos] = True
                                    history_local_position_ids[index][pos] = i
                                    pos += 1
                                break
                        max_history_length = max(max_history_length, pos)
            with open(history_input_ids_file, 'wb') as f:
                pickle.dump(history_input_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(history_segment_ids_file, 'wb') as f:
                pickle.dump(history_segment_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(history_global_attention_mask_file, 'wb') as f:
                pickle.dump(history_global_attention_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(history_local_position_ids_file, 'wb') as f:
                pickle.dump(history_local_position_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            if mode != 'train':
                num = len(quote_map)
                quote_indices = info[mode + '_quote_indices']
                with open(os.path.join(dataset_root, 'truth-' + mode + '.txt'), 'w', encoding='utf-8') as f:
                    for i, quote_index in enumerate(quote_indices):
                        labels = [0 for _ in range(num)]
                        labels[quote_index] = 1
                        f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(labels).replace(' ', ''))
        print(dataset_root, 'max_history_length =', max_history_length)


def debug(encoder_max_lengths, decoder_max_lengths):
    debug_index = 128
    for dataset in ['Reddit-quote', 'QuoteR']:
        with open(dataset + '.debug', 'w', encoding='utf-8') as debug_f:
            dataset_root = 'quoteRec/' + dataset
            encoder_max_length = encoder_max_lengths[dataset_root]
            decoder_max_length = decoder_max_lengths[dataset_root]
            for mode in ['train', 'test']:
                with open(os.path.join(dataset_root, 'history_input_ids-%s-%d.pkl' % (mode, encoder_max_length)), 'rb') as f:
                    history_input_ids = pickle.load(f)[debug_index]
                with open(os.path.join(dataset_root, 'history_segment_ids-%s-%d.pkl' % (mode, encoder_max_length)), 'rb') as f:
                    history_segment_ids = pickle.load(f)[debug_index]
                with open(os.path.join(dataset_root, 'history_global_attention_mask-%s-%d.pkl' % (mode, encoder_max_length)), 'rb') as f:
                    history_global_attention_mask = pickle.load(f)[debug_index]
                with open(os.path.join(dataset_root, 'history_local_position_ids-%s-%d.pkl' % (mode, encoder_max_length)), 'rb') as f:
                    history_local_position_ids = pickle.load(f)[debug_index]
                debug_f.write(dataset + '\t' + mode + '\tuser history :\n')
                segment_num = 0
                for i in range(encoder_max_length):
                    if history_segment_ids[i] != MAX_HISTORY_NUM:
                        segment_num = max(history_segment_ids[i], segment_num)
                for i in range(1, segment_num + 1):
                    input_ids = []
                    for j in range(encoder_max_length):
                        if history_segment_ids[j] == i:
                            assert history_global_attention_mask[j]
                            assert history_local_position_ids[j] == len(input_ids)
                            input_ids.append(history_input_ids[j])
                    text = tokenizer.decode(input_ids)
                    debug_f.write(text + '\n')
                assert history_input_ids[0] == BOS_TOKEN_ID
                assert history_global_attention_mask[0]
                for i in range(1, encoder_max_length):
                    if history_segment_ids[i] > segment_num:
                        assert history_segment_ids[i] == MAX_HISTORY_NUM
                        assert (history_input_ids[i] == EOS_TOKEN_ID) == history_global_attention_mask[i]
                debug_f.write('\n')
            with open(os.path.join(dataset_root, 'quote_input_ids-%d.pkl' % decoder_max_length), 'rb') as f:
                quote_input_ids = pickle.load(f)[debug_index]
            with open(os.path.join(dataset_root, 'quote_cls_indices-%d.pkl' % decoder_max_length), 'rb') as f:
                quote_cls_index = pickle.load(f)[debug_index]
            with open(os.path.join(dataset_root, 'quote_attention_mask-%d.pkl' % decoder_max_length), 'rb') as f:
                quote_attention_mask = pickle.load(f)[debug_index]
            debug_f.write('\n' + dataset + '\tquote :\n')
            assert quote_input_ids[quote_cls_index] == EOS_TOKEN_ID
            input_ids = []
            for i in range(decoder_max_length):
                if quote_attention_mask[i]:
                    input_ids.append(quote_input_ids[i])
            text = tokenizer.decode(input_ids)
            debug_f.write(text + '\n\n')


if __name__ == '__main__':
    encoder_max_lengths = {
        reddit_quote_dataset_root: 1024,
        quoter_dataset_root: 384
    }
    decoder_max_lengths = {
        reddit_quote_dataset_root: 72,
        quoter_dataset_root: 84
    }
    download_QuoteDatasets()
    preprocess_datasets(encoder_max_lengths, decoder_max_lengths)
    debug(encoder_max_lengths, decoder_max_lengths)
