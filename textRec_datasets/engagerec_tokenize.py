import sys
sys.path.append('..')
import os
from collections import Counter
import json
import pickle
import numpy as np
import re
from transformers.models.bart import BartTokenizer
BOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
DECODER_START_TOKEN_ID = 2 # a unique hack of BART, also refer to https://github.com/huggingface/transformers/issues/5212
IGNORE_TOKEN_ID = -100
MAX_HISTORY_NUM = 100
train_HISTORY_RATIO = 0.2
engagerec_root = 'engageRec'
if not os.path.exists(engagerec_root):
    os.mkdir(engagerec_root)
reddit_engage_dataset_roots = {
    'funny': os.path.join(engagerec_root, 'Reddit-engage_funny'),
    'technology': os.path.join(engagerec_root, 'Reddit-engage_technology'),
    'todayilearned': os.path.join(engagerec_root, 'Reddit-engage_todayilearned')
}
for reddit_engage_dataset_root in reddit_engage_dataset_roots.values():
    if not os.path.exists(reddit_engage_dataset_root):
        os.mkdir(reddit_engage_dataset_root)
if not os.path.exists('../../download'):
    os.mkdir('../../download')
if not os.path.exists('../../backbone_models'):
    os.mkdir('../../backbone_models')
if not os.path.exists('../../backbone_models/bart-base'):
    os.system('git lfs install')
    os.system('git clone https://huggingface.co/facebook/bart-base ../../backbone_models/bart-base')
tokenizer = BartTokenizer.from_pretrained('../../backbone_models/bart-base')


def download_engageDatasets():
    for prefix in ['funny', 'technology', 'todayilearned']:
        for i in range(4):
            if not os.path.exists('../../download/%s_20150%d.data' % (prefix, i + 1)):
                os.system('wget https://github.com/zxshamson/dy-conv-rec/raw/master/Datafiles/%s_20150%d.data -P ../../download' % (prefix, i + 1))
        if not os.path.exists('../../download/%s_201505_valid.data' % prefix):
            os.system('wget https://github.com/zxshamson/dy-conv-rec/raw/master/Datafiles/%s_201505_valid.data -P ../../download' % prefix)
        if not os.path.exists('../../download/%s_201505_test.data' % prefix):
            os.system('wget https://github.com/zxshamson/dy-conv-rec/raw/master/Datafiles/%s_201505_test.data -P ../../download' % prefix)
        train_file = os.path.join(reddit_engage_dataset_roots[prefix], 'train.json')
        dev_file = os.path.join(reddit_engage_dataset_roots[prefix], 'dev.json')
        test_file = os.path.join(reddit_engage_dataset_roots[prefix], 'test.json')
        conv_map_file = os.path.join(reddit_engage_dataset_roots[prefix], 'conv_map.json')
        msg_map_file = os.path.join(reddit_engage_dataset_roots[prefix], 'msg_map.json')
        user_map_file = os.path.join(reddit_engage_dataset_roots[prefix], 'user_map.json')
        msg_token_file = os.path.join(reddit_engage_dataset_roots[prefix], 'msg_tokens.pkl')
        msg_conv_file = os.path.join(reddit_engage_dataset_roots[prefix], 'msg_conv.json')
        user_msg_file = os.path.join(reddit_engage_dataset_roots[prefix], 'user_msg.json')
        if all(list(map(os.path.exists, [train_file, dev_file, test_file, conv_map_file, msg_map_file, user_map_file, msg_token_file, msg_conv_file, user_msg_file]))):
            continue
        train_map, dev_map, test_map = {}, {}, {}
        conv_map, msg_map, user_map = {}, {}, {}
        msg_tokens = []
        msg_conv, user_msg = {}, {}
        substitute_msg, invalid_conv = {}, set()
        max_decoder_length, cnt1, cnt2 = 0, 0, 0

        with open(train_file, 'w', encoding='utf-8') as f:
            visited_msg = set()
            for i in range(4):
                with open('../../download/%s_20150%d.data' % (prefix, i + 1), 'r', encoding='utf-8') as f_:
                    for line in f_:
                        line = line.strip()
                        if len(line) > 0:
                            conv_id, msg_id, parent_id, post, post_, user_id, time, up, down = line.split('\t')
                            conv_id += '-train'
                            _id = conv_id + msg_id
                            if _id in visited_msg:
                                continue
                            visited_msg.add(_id)
                            if conv_id in invalid_conv:
                                continue
                            assert conv_id[:3] != '05_'
                            post = post.strip()
                            if '<URL>' in post_:
                                post = post.replace('Http://', 'http://').replace('Https://', 'https://').replace('Www.', 'www.')
                                post = re.sub('(?P<url>http?://[^\s]+)', 'URL', post)
                                post = re.sub('(?P<url>https?://[^\s]+)', 'URL', post)
                                post = re.sub('(?P<url>www.[^\s]+)', 'URL', post)
                                post = re.sub('(?P<url>[^\s]+).com', 'URL', post)
                            post = post.replace(' &amp; ', ' and ')
                            post = post.replace('&amp;', ' and ')
                            post = post.replace('&lt;', '<')
                            post = post.replace('&gt;', '>')
                            post = post.replace('*', '')
                            post = post.replace('~', '')
                            while '^^' in post:
                                post = post.replace('^^', '^')
                            while '  ' in post:
                                post = post.replace('  ', ' ')
                            while '__' in post:
                                post = post.replace('__', '_')
                            while '--' in post:
                                post = post.replace('--', '-')
                            while '# #' in post:
                                post = post.replace('# #', '#')
                            while '##' in post:
                                post = post.replace('##', '#')
                            while '....' in post:
                                post = post.replace('....', '...')
                            if len(post) == 0:
                                if parent_id == 'null':
                                    assert conv_id not in conv_map
                                    invalid_conv.add(conv_id)
                                    continue
                                substitute_msg[msg_id] = parent_id
                                assert parent_id not in substitute_msg
                                continue
                            if conv_id not in conv_map:
                                conv_map[conv_id] = len(conv_map)
                            assert msg_id not in msg_map
                            msg_map[msg_id] = len(msg_map)
                            while '  ' in post:
                                post = post.replace('  ', ' ')
                            msg_tokens.append(tokenizer(post, add_special_tokens=False)['input_ids'] + [EOS_TOKEN_ID])
                            if user_id not in user_map:
                                user_map[user_id] = len(user_map)
                            user_id = user_map[user_id]
                            conv_id = conv_map[conv_id]
                            msg_id = msg_map[msg_id]
                            if conv_id not in train_map:
                                train_map[conv_id] = [[msg_id, msg_map[parent_id] if parent_id != 'null' else -1, user_id, int(time)]]
                            else:
                                if parent_id not in substitute_msg:
                                    train_map[conv_id].append([msg_id, msg_map[parent_id] if parent_id != 'null' else -1, user_id, int(time)])
                                else:
                                    train_map[conv_id].append([msg_id, msg_map[substitute_msg[parent_id]] if parent_id != 'null' else -1, user_id, int(time)])
                            msg_conv[msg_id] = conv_id
                            if user_id not in user_msg:
                                user_msg[user_id] = [[msg_id, int(time)]]
                            else:
                                user_msg[user_id].append([msg_id, int(time)])
            for conv_id in train_map:
                threads = list(map(lambda x: x[:-1], sorted(train_map[conv_id], key=lambda x: x[-1])))
                train_map[conv_id] = threads
                decoder_length = 1
                for i, thread in enumerate(threads):
                    assert (thread[1] == -1) == (i == 0)
                    decoder_length += len(msg_tokens[thread[0]]) + 1
                if decoder_length <= 576:
                    cnt1 += 1
                else:
                    cnt2 += 1
                max_decoder_length = max(decoder_length, max_decoder_length)
            json.dump(train_map, f)
        with open(msg_conv_file, 'w', encoding='utf-8') as f:
            json.dump(msg_conv, f)
        for user_id in user_msg:
            user_msg[user_id] = list(map(lambda x: x[0], sorted(user_msg[user_id], key=lambda x: x[1])))
        with open(user_msg_file, 'w', encoding='utf-8') as f:
            json.dump(user_msg, f)
        with open(dev_file, 'w', encoding='utf-8') as f:
            conv_cnt = Counter()
            with open('../../download/%s_201505_valid.data' % prefix, 'r', encoding='utf-8') as f_:
                for line in f_:
                    line = line.strip()
                    if len(line) > 0:
                        data = line.split('\t')
                    if len(data) == 9:
                        conv_id, msg_id, parent_id, post, post_, user_id, time, up, down = data
                        conv_cnt[conv_id] += 1
            with open('../../download/%s_201505_valid.data' % prefix, 'r', encoding='utf-8') as f_:
                visited_msg = set()
                for line in f_:
                    line = line.strip()
                    if len(line) > 0:
                        data = line.split('\t')
                        if len(data) == 9:
                            conv_id, msg_id, parent_id, post, post_, user_id, time, up, down = data
                            if conv_cnt[conv_id] > 1:
                                continue # since the orignal Reddit Enagement dataset does not label which post the users engage with, we can only use the first post
                            conv_id += '-dev'
                            _id = conv_id + msg_id
                            if _id in visited_msg:
                                continue
                            visited_msg.add(_id)
                            if conv_id in invalid_conv:
                                continue
                            post = post.strip()
                            if '<URL>' in post_:
                                post = post.replace('Http://', 'http://').replace('Https://', 'https://').replace('Www.', 'www.')
                                post = re.sub('(?P<url>http?://[^\s]+)', 'URL', post)
                                post = re.sub('(?P<url>https?://[^\s]+)', 'URL', post)
                                post = re.sub('(?P<url>www.[^\s]+)', 'URL', post)
                                post = re.sub('(?P<url>[^\s]+).com', 'URL', post)
                            post = post.replace(' &amp; ', ' and ')
                            post = post.replace('&amp;', ' and ')
                            post = post.replace('&lt;', '<')
                            post = post.replace('&gt;', '>')
                            post = post.replace('*', '')
                            post = post.replace('~', '')
                            while '^^' in post:
                                post = post.replace('^^', '^')
                            while '  ' in post:
                                post = post.replace('  ', ' ')
                            while '__' in post:
                                post = post.replace('__', '_')
                            while '--' in post:
                                post = post.replace('--', '-')
                            while '# #' in post:
                                post = post.replace('# #', '#')
                            while '##' in post:
                                post = post.replace('##', '#')
                            while '....' in post:
                                post = post.replace('....', '...')
                            if len(post) == 0:
                                if parent_id == 'null':
                                    assert conv_id not in conv_map
                                    invalid_conv.add(conv_id)
                                    continue
                                substitute_msg[msg_id] = parent_id
                                assert parent_id not in substitute_msg
                                continue
                            if conv_id not in conv_map:
                                conv_map[conv_id] = len(conv_map)
                            if msg_id not in msg_map:
                                msg_map[msg_id] = len(msg_map)
                                while '  ' in post:
                                    post = post.replace('  ', ' ')
                                msg_tokens.append(tokenizer(post, add_special_tokens=False)['input_ids'] + [EOS_TOKEN_ID])
                            if user_id not in user_map:
                                user_map[user_id] = len(user_map)
                            user_id = user_map[user_id]
                            conv_id = conv_map[conv_id]
                            msg_id = msg_map[msg_id]
                            if conv_id not in dev_map:
                                dev_map[conv_id] = {'threads': [[msg_id, msg_map[parent_id] if parent_id != 'null' else -1, user_id, int(time)]]}
                            else:
                                if parent_id not in substitute_msg:
                                    dev_map[conv_id]['threads'].append([msg_id, msg_map[parent_id] if parent_id != 'null' else -1, user_id, int(time)])
                                else:
                                    dev_map[conv_id]['threads'].append([msg_id, msg_map[substitute_msg[parent_id]] if parent_id != 'null' else -1, user_id, int(time)])
                        else:
                            conv_id, user_ids = data
                            if conv_cnt[conv_id] > 1:
                                continue
                            conv_id += '-dev'
                            if conv_id in invalid_conv:
                                continue
                            assert conv_id in conv_map and 'users' not in dev_map[conv_map[conv_id]]
                            for i, user_id in enumerate(user_ids.strip().split(' ')):
                                if user_id not in user_map:
                                    user_map[user_id] = len(user_map)
                                user_id = user_map[user_id]
                                if i == 0:
                                    dev_map[conv_map[conv_id]]['users'] = [user_id]
                                else:
                                    if user_id not in dev_map[conv_map[conv_id]]['users']:
                                        dev_map[conv_map[conv_id]]['users'].append(user_id)
            for conv_id in dev_map:
                threads = list(map(lambda x: x[:-1], sorted(dev_map[conv_id]['threads'], key=lambda x: x[-1])))
                dev_map[conv_id]['threads'] = threads
                decoder_length = 1
                for i, thread in enumerate(threads):
                    assert (thread[1] == -1) == (i == 0)
                    decoder_length += len(msg_tokens[thread[0]]) + 1
                if decoder_length <= 576:
                    cnt1 += 1
                else:
                    cnt2 += 1
                max_decoder_length = max(decoder_length, max_decoder_length)
            json.dump(dev_map, f)
        with open(test_file, 'w', encoding='utf-8') as f:
            conv_cnt = Counter()
            with open('../../download/%s_201505_test.data' % prefix, 'r', encoding='utf-8') as f_:
                for line in f_:
                    line = line.strip()
                    if len(line) > 0:
                        data = line.split('\t')
                    if len(data) == 9:
                        conv_id, msg_id, parent_id, post, post_, user_id, time, up, down = data
                        conv_cnt[conv_id] += 1
            with open('../../download/%s_201505_test.data' % prefix, 'r', encoding='utf-8') as f_:
                visited_msg = set()
                for line in f_:
                    line = line.strip()
                    if len(line) > 0:
                        data = line.split('\t')
                        if len(data) == 9:
                            conv_id, msg_id, parent_id, post, post_, user_id, time, up, down = data
                            if conv_cnt[conv_id] > 1:
                                continue # since the orignal Reddit Enagement dataset does not label which post the users engage with, we can only use the first post
                            conv_id += '-test'
                            _id = conv_id + msg_id
                            if _id in visited_msg:
                                continue
                            visited_msg.add(_id)
                            if conv_id in invalid_conv:
                                continue
                            post = post.strip()
                            if '<URL>' in post_:
                                post = post.replace('Http://', 'http://').replace('Https://', 'https://').replace('Www.', 'www.')
                                post = re.sub('(?P<url>http?://[^\s]+)', 'URL', post)
                                post = re.sub('(?P<url>https?://[^\s]+)', 'URL', post)
                                post = re.sub('(?P<url>www.[^\s]+)', 'URL', post)
                                post = re.sub('(?P<url>[^\s]+).com', 'URL', post)
                            post = post.replace(' &amp; ', ' and ')
                            post = post.replace('&amp;', ' and ')
                            post = post.replace('&lt;', '<')
                            post = post.replace('&gt;', '>')
                            post = post.replace('*', '')
                            post = post.replace('~', '')
                            while '^^' in post:
                                post = post.replace('^^', '^')
                            while '  ' in post:
                                post = post.replace('  ', ' ')
                            while '__' in post:
                                post = post.replace('__', '_')
                            while '--' in post:
                                post = post.replace('--', '-')
                            while '# #' in post:
                                post = post.replace('# #', '#')
                            while '##' in post:
                                post = post.replace('##', '#')
                            while '....' in post:
                                post = post.replace('....', '...')
                            if len(post) == 0:
                                if parent_id == 'null':
                                    assert conv_id not in conv_map
                                    invalid_conv.add(conv_id)
                                    continue
                                substitute_msg[msg_id] = parent_id
                                assert parent_id not in substitute_msg
                                continue
                            if conv_id not in conv_map:
                                conv_map[conv_id] = len(conv_map)
                            if msg_id not in msg_map:
                                msg_map[msg_id] = len(msg_map)
                                while '  ' in post:
                                    post = post.replace('  ', ' ')
                                msg_tokens.append(tokenizer(post, add_special_tokens=False)['input_ids'] + [EOS_TOKEN_ID])
                            if user_id not in user_map:
                                user_map[user_id] = len(user_map)
                            user_id = user_map[user_id]
                            conv_id = conv_map[conv_id]
                            msg_id = msg_map[msg_id]
                            if conv_id not in test_map:
                                test_map[conv_id] = {'threads': [[msg_id, msg_map[parent_id] if parent_id != 'null' else -1, user_id, int(time)]]}
                            else:
                                if parent_id not in substitute_msg:
                                    test_map[conv_id]['threads'].append([msg_id, msg_map[parent_id] if parent_id != 'null' else -1, user_id, int(time)])
                                else:
                                    test_map[conv_id]['threads'].append([msg_id, msg_map[substitute_msg[parent_id]] if parent_id != 'null' else -1, user_id, int(time)])
                        else:
                            conv_id, user_ids = data
                            if conv_cnt[conv_id] > 1:
                                continue
                            conv_id += '-test'
                            if conv_id in invalid_conv:
                                continue
                            assert conv_id in conv_map and 'users' not in test_map[conv_map[conv_id]]
                            for i, user_id in enumerate(user_ids.strip().split(' ')):
                                if user_id not in user_map:
                                    user_map[user_id] = len(user_map)
                                user_id = user_map[user_id]
                                if i == 0:
                                    test_map[conv_map[conv_id]]['users'] = [user_id]
                                else:
                                    if user_id not in test_map[conv_map[conv_id]]['users']:
                                        test_map[conv_map[conv_id]]['users'].append(user_id)
            for conv_id in test_map:
                threads = list(map(lambda x: x[:-1], sorted(test_map[conv_id]['threads'], key=lambda x: x[-1])))
                test_map[conv_id]['threads'] = threads
                decoder_length = 1
                for i, thread in enumerate(threads):
                    assert (thread[1] == -1) == (i == 0)
                    decoder_length += len(msg_tokens[thread[0]]) + 1
                if decoder_length <= 576:
                    cnt1 += 1
                else:
                    cnt2 += 1
                max_decoder_length = max(decoder_length, max_decoder_length)
            json.dump(test_map, f)

        assert 'null' not in msg_map
        with open(conv_map_file, 'w', encoding='utf-8') as f:
            json.dump(conv_map, f)
        with open(msg_map_file, 'w', encoding='utf-8') as f:
            json.dump(msg_map, f)
        with open(user_map_file, 'w', encoding='utf-8') as f:
            json.dump(user_map, f)
        with open(msg_token_file, 'wb') as f:
            pickle.dump(msg_tokens, f)
        print(prefix, ':', len(conv_map), len(msg_map), len(user_map), ' | ', max_decoder_length, cnt1, cnt2)


def prepreprocess():
    def convert_json_map(json_map):
        return {int(k): json_map[k] for k in json_map}
    for prefix in ['funny', 'technology', 'todayilearned']:
        with open(os.path.join(reddit_engage_dataset_roots[prefix], 'train.json'), 'r', encoding='utf-8') as f:
            train_map = convert_json_map(json.load(f))
        with open(os.path.join(reddit_engage_dataset_roots[prefix], 'dev.json'), 'r', encoding='utf-8') as f:
            dev_map = convert_json_map(json.load(f))
        with open(os.path.join(reddit_engage_dataset_roots[prefix], 'test.json'), 'r', encoding='utf-8') as f:
            test_map = convert_json_map(json.load(f))
        with open(os.path.join(reddit_engage_dataset_roots[prefix], 'msg_conv.json'), 'r', encoding='utf-8') as f:
            msg_conv = convert_json_map(json.load(f))
        with open(os.path.join(reddit_engage_dataset_roots[prefix], 'user_msg.json'), 'r', encoding='utf-8') as f:
            user_msg = convert_json_map(json.load(f))
        with open(os.path.join(reddit_engage_dataset_roots[prefix], 'msg_tokens.pkl'), 'rb') as f:
            msg_tokens = pickle.load(f)
        conv_token_file = os.path.join(reddit_engage_dataset_roots[prefix], 'conv.pkl') # for all conv tokens
        turn_token_file = os.path.join(reddit_engage_dataset_roots[prefix], 'turn.pkl') # for train history tokens
        # 1. conv and turn tokens
        if not os.path.exists(conv_token_file) or not os.path.exists(turn_token_file):
            conv_tokens = []
            turn_tokens = [None for _ in range(len(msg_conv))]
            for index, data_map in enumerate([train_map, dev_map, test_map]):
                for conv_id in data_map:
                    data = data_map[conv_id] if index == 0 else data_map[conv_id]['threads']
                    assert conv_id == len(conv_tokens)
                    tokens = []
                    parent_map = {}
                    for msg_id, parent_id, user_id in data:
                        tokens.append(msg_tokens[msg_id])
                        parent_map[msg_id] = parent_id
                    if index == 0:
                        for msg_id in parent_map:
                            if turn_tokens[msg_id] is None:
                                thread_tokens = [msg_id]
                                _msg_id = parent_map[msg_id]
                                while _msg_id != -1:
                                    thread_tokens.append(_msg_id)
                                    _msg_id = parent_map[_msg_id]
                                if len(thread_tokens) == 1:
                                    turn_tokens[msg_id] = [msg_tokens[thread_tokens[0]]]
                                elif len(thread_tokens) == 2:
                                    turn_tokens[msg_id] = [msg_tokens[thread_tokens[1]], msg_tokens[thread_tokens[0]]]
                                else:
                                    turn_tokens[msg_id] = [msg_tokens[thread_tokens[-1]], msg_tokens[thread_tokens[1]], msg_tokens[thread_tokens[0]]]
                    conv_tokens.append(tokens)
            assert all(list(map(lambda x: x is not None, turn_tokens)))
            with open(conv_token_file, 'wb') as f:
                pickle.dump(conv_tokens, f)
            with open(turn_token_file, 'wb') as f:
                pickle.dump(turn_tokens, f)
        else:
            with open(conv_token_file, 'rb') as f:
                conv_tokens = pickle.load(f)
            with open(turn_token_file, 'rb') as f:
                turn_tokens = pickle.load(f)
        # 2. candidate pools and meta data
        train_candidate_file = os.path.join(reddit_engage_dataset_roots[prefix], 'train_candidates.pkl')
        train_meta_file = os.path.join(reddit_engage_dataset_roots[prefix], 'train_meta.pkl')
        dev_candidate_file = os.path.join(reddit_engage_dataset_roots[prefix], 'dev_candidates.pkl')
        dev_meta_file = os.path.join(reddit_engage_dataset_roots[prefix], 'dev_meta.pkl')
        test_candidate_file = os.path.join(reddit_engage_dataset_roots[prefix], 'test_candidates.pkl')
        test_meta_file = os.path.join(reddit_engage_dataset_roots[prefix], 'test_meta.pkl')
        if all(list(map(os.path.exists, [train_candidate_file, train_meta_file, dev_candidate_file, dev_meta_file, test_candidate_file, test_meta_file]))):
            continue
        train_candidates, dev_candidates, test_candidates = [], [], []
        train_meta, dev_meta, test_meta = {}, {}, {}
        _dev_map = {}
        for conv_id in dev_map:
            user_ids = []
            for user_id in dev_map[conv_id]['users']:
                if user_id in user_msg:
                    user_ids.append(user_id)
            if len(user_ids) > 0:
                _dev_map[conv_id] = user_ids
        _test_map = {}
        for conv_id in test_map:
            user_ids = []
            for user_id in test_map[conv_id]['users']:
                if user_id in user_msg:
                    user_ids.append(user_id)
            if len(user_ids) > 0:
                _test_map[conv_id] = user_ids
        for user_id in user_msg:
            if len(user_msg[user_id]) > 1:
                msg_ids = user_msg[user_id]
                N = max(int(len(msg_ids) * train_HISTORY_RATIO), 1)
                history_msg_ids = msg_ids[:-N]
                candidate_msg_ids = msg_ids[-N:]
                history = [turn_tokens[msg_id] for msg_id in history_msg_ids]
                candidate_ids = []
                for msg_id in candidate_msg_ids:
                    candidate = []
                    conv_id = msg_conv[msg_id]
                    while True:
                        for _msg_id, parent_id, _user_id in train_map[conv_id]:
                            if _msg_id == msg_id:
                                candidate.append(msg_tokens[msg_id])
                                break
                        if parent_id == -1:
                            break
                        msg_id = parent_id
                    candidate = candidate[1:]
                    if len(candidate) == 0:
                        continue
                    candidate.reverse()
                    candidate_ids.append(len(train_candidates))
                    train_candidates.append(candidate)
                if len(candidate_ids) > 0:
                    train_meta[user_id] = [history, candidate_ids]
        with open(train_candidate_file, 'wb') as f:
            pickle.dump(train_candidates, f)
        with open(train_meta_file, 'wb') as f:
            pickle.dump(train_meta, f)
        for conv_id, user_ids in _dev_map.items():
            candidate_id = len(dev_candidates)
            for user_id in user_ids:
                if user_id not in dev_meta:
                    dev_meta[user_id] = [[turn_tokens[msg_id] for msg_id in user_msg[user_id]], [candidate_id]]
                else:
                    dev_meta[user_id][1].append(candidate_id)
            dev_candidates.append(conv_tokens[conv_id])
        with open(dev_candidate_file, 'wb') as f:
            pickle.dump(dev_candidates, f)
        with open(dev_meta_file, 'wb') as f:
            pickle.dump(dev_meta, f)
        for conv_id, user_ids in _test_map.items():
            candidate_id = len(test_candidates)
            for user_id in user_ids:
                if user_id not in test_meta:
                    test_meta[user_id] = [[turn_tokens[msg_id] for msg_id in user_msg[user_id]], [candidate_id]]
                else:
                    test_meta[user_id][1].append(candidate_id)
            test_candidates.append(conv_tokens[conv_id])
        with open(test_candidate_file, 'wb') as f:
            pickle.dump(test_candidates, f)
        with open(test_meta_file, 'wb') as f:
            pickle.dump(test_meta, f)


def debug_prepreprocess():
    debug_index = 256
    with open('engagerec.debug', 'w', encoding='utf-8') as debug_f:
        for prefix in ['funny', 'technology', 'todayilearned']:
            user_map_file = os.path.join(reddit_engage_dataset_roots[prefix], 'user_map.json')
            train_candidate_file = os.path.join(reddit_engage_dataset_roots[prefix], 'train_candidates.pkl')
            train_meta_file = os.path.join(reddit_engage_dataset_roots[prefix], 'train_meta.pkl')
            dev_candidate_file = os.path.join(reddit_engage_dataset_roots[prefix], 'dev_candidates.pkl')
            dev_meta_file = os.path.join(reddit_engage_dataset_roots[prefix], 'dev_meta.pkl')
            test_candidate_file = os.path.join(reddit_engage_dataset_roots[prefix], 'test_candidates.pkl')
            test_meta_file = os.path.join(reddit_engage_dataset_roots[prefix], 'test_meta.pkl')
            with open(user_map_file, 'r', encoding='utf-8') as f:
                user_map = json.load(f)
                user_map = {v:k for k, v in user_map.items()}
            with open(train_candidate_file, 'rb') as f:
                train_candidates = pickle.load(f)
            with open(train_meta_file, 'rb') as f:
                train_meta = pickle.load(f)
            with open(dev_candidate_file, 'rb') as f:
                dev_candidates = pickle.load(f)
            with open(dev_meta_file, 'rb') as f:
                dev_meta = pickle.load(f)
            with open(test_candidate_file, 'rb') as f:
                test_candidates = pickle.load(f)
            with open(test_meta_file, 'rb') as f:
                test_meta = pickle.load(f)
            for index, user_id in enumerate(train_meta):
                if index == debug_index:
                    debug_f.write('%s : train\nDebug user : %s\n' % (prefix, user_map[user_id]))
                    history, candidates = train_meta[user_id]
                    for i, history_data in enumerate(history):
                        for j, tokens in enumerate(history_data):
                            debug_f.write('[%d, %d] : %s\n' % (i, j, tokenizer.decode(tokens)))
                    debug_f.write('Candidates\n')
                    for i, candidate in enumerate(candidates):
                        for j, turn in enumerate(train_candidates[candidate]):
                            debug_f.write('[%d, %d] : %s\n' % (i, j, tokenizer.decode(turn)))
                    break
            for index, user_id in enumerate(dev_meta):
                if index == debug_index:
                    debug_f.write('%s : dev\nDebug user : %s\n' % (prefix, user_map[user_id]))
                    history, candidates = dev_meta[user_id]
                    for i, history_data in enumerate(history):
                        for j, tokens in enumerate(history_data):
                            debug_f.write('[%d, %d] : %s\n' % (i, j, tokenizer.decode(tokens)))
                    debug_f.write('Candidates\n')
                    for i, candidate in enumerate(candidates):
                        for j, turn in enumerate(dev_candidates[candidate]):
                            debug_f.write('[%d, %d] : %s\n' % (i, j, tokenizer.decode(turn)))
                    break
            for index, user_id in enumerate(test_meta):
                if index == debug_index:
                    debug_f.write('%s : test\nDebug user : %s\n' % (prefix, user_map[user_id]))
                    history, candidates = test_meta[user_id]
                    for i, history_data in enumerate(history):
                        for j, tokens in enumerate(history_data):
                            debug_f.write('[%d, %d] : %s\n' % (i, j, tokenizer.decode(tokens)))
                    debug_f.write('Candidates\n')
                    for i, candidate in enumerate(candidates):
                        for j, turn in enumerate(test_candidates[candidate]):
                            debug_f.write('[%d, %d] : %s\n' % (i, j, tokenizer.decode(turn)))
                    break


def preprocess_history_inputs(max_length):
    for prefix in ['funny', 'technology', 'todayilearned']:
        for mode in ['train', 'dev', 'test']:
            history_input_ids_file = os.path.join(reddit_engage_dataset_roots[prefix], 'history_input_ids-%s-%d.pkl' % (mode, max_length))
            history_segment_ids_file = os.path.join(reddit_engage_dataset_roots[prefix], 'history_segment_ids-%s-%d.pkl' % (mode, max_length))
            history_global_attention_mask_file = os.path.join(reddit_engage_dataset_roots[prefix], 'history_global_attention_mask-%s-%d.pkl' % (mode, max_length))
            history_local_position_ids_file = os.path.join(reddit_engage_dataset_roots[prefix], 'history_local_position_ids-%s-%d.pkl' % (mode, max_length))
            history_user_map_file = os.path.join(reddit_engage_dataset_roots[prefix], 'history_user_map-%s.json' % mode)
            val_flag = mode in ['dev', 'test']
            if all(list(map(os.path.exists, [history_input_ids_file, history_segment_ids_file, history_global_attention_mask_file, history_local_position_ids_file, history_user_map_file]))):
                if mode == 'train' or os.path.exists(os.path.join(reddit_engage_dataset_roots[prefix], 'truth-%s.txt' % mode)):
                    continue
            with open(os.path.join(reddit_engage_dataset_roots[prefix], mode + '_meta.pkl'), 'rb') as f:
                meta_data = pickle.load(f)
            if val_flag:
                truth_f = open(os.path.join(reddit_engage_dataset_roots[prefix], 'truth-%s.txt' % mode), 'w', encoding='utf-8')
                with open(os.path.join(reddit_engage_dataset_roots[prefix], mode + '_candidates.pkl'), 'rb') as f:
                    candidate_num = len(pickle.load(f))
            num = len(meta_data)
            history_input_ids = np.full([num, max_length], PAD_TOKEN_ID, dtype=np.uint16)
            history_segment_ids = np.full([num, max_length], MAX_HISTORY_NUM, dtype=np.uint8)
            history_global_attention_mask = np.zeros([num, max_length], dtype=bool)
            history_local_position_ids = np.zeros([num, max_length], dtype=np.int16)
            history_input_ids[:, 0] = BOS_TOKEN_ID
            history_segment_ids[:, 0] = 0
            history_global_attention_mask[:, 0] = True
            N, length, cnt1, cnt2 = 0, 0, 0, 0
            history_user_map = {}
            for index, user_id in enumerate(meta_data):
                history_turns, candidates = meta_data[user_id]
                if val_flag:
                    labels = [0 for _ in range(candidate_num)]
                    for candidate in candidates:
                        labels[candidate] = 1
                    truth_f.write(('' if index == 0 else '\n') + str(index + 1) + ' ' + str(labels).replace(' ', ''))
                N += len(candidates)
                history_turns.reverse()
                pos = 1
                flag = True
                for history_index, turn in enumerate(history_turns):
                    history_tokens = []
                    for tokens in turn:
                        history_tokens.extend(tokens)
                    history_token_num = len(history_tokens)
                    if pos + history_token_num < max_length:
                        for i in range(history_token_num):
                            history_input_ids[index][pos] = history_tokens[i]
                            history_segment_ids[index][pos] = history_index + 1
                            history_global_attention_mask[index][pos] = True
                            history_local_position_ids[index][pos] = i
                            pos += 1
                    else:
                        for i in range(max_length - pos):
                            history_input_ids[index][pos] = history_tokens[i]
                            history_segment_ids[index][pos] = history_index + 1
                            history_global_attention_mask[index][pos] = True
                            history_local_position_ids[index][pos] = i
                            pos += 1
                        flag = False
                        break
                if flag:
                    cnt1 += 1
                else:
                    cnt2 += 1
                length += pos
                history_user_map[user_id] = index
            with open(history_input_ids_file, 'wb') as f:
                pickle.dump(history_input_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(history_segment_ids_file, 'wb') as f:
                pickle.dump(history_segment_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(history_global_attention_mask_file, 'wb') as f:
                pickle.dump(history_global_attention_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(history_local_position_ids_file, 'wb') as f:
                pickle.dump(history_local_position_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(history_user_map_file, 'w', encoding='utf-8') as f:
                json.dump(history_user_map, f)
            if val_flag:
                truth_f.close()
            print(prefix, mode, N, length / num, cnt1, cnt2)


def preprocess_candidate_inputs(max_length):
    for prefix in ['funny', 'technology', 'todayilearned']:
        for mode in ['train', 'dev', 'test']:
            engage_input_ids_file = os.path.join(reddit_engage_dataset_roots[prefix], 'engage_input_ids-%s-%d.pkl' % (mode, max_length))
            engage_cls_indices_file = os.path.join(reddit_engage_dataset_roots[prefix], 'engage_cls_indices-%s-%d.pkl' % (mode, max_length))
            engage_attention_mask_file = os.path.join(reddit_engage_dataset_roots[prefix], 'engage_attention_mask-%s-%d.pkl' % (mode, max_length))
            targets_file = os.path.join(reddit_engage_dataset_roots[prefix], 'targets-%s-%d.pkl' % (mode, max_length))
            if all(list(map(os.path.exists, [engage_input_ids_file, engage_cls_indices_file, engage_attention_mask_file, targets_file]))):
                continue
            with open(os.path.join(reddit_engage_dataset_roots[prefix], mode + '_candidates.pkl'), 'rb') as f:
                candidate_data = pickle.load(f)
            num = len(candidate_data)
            engage_input_ids = np.full([num, max_length], PAD_TOKEN_ID, dtype=np.int32)
            engage_cls_indices = np.zeros([num], dtype=np.int32)
            engage_attention_mask = np.zeros([num, max_length], dtype=bool)
            targets = np.full([num, max_length], PAD_TOKEN_ID, dtype=np.int64)
            engage_input_ids[:, 0] = DECODER_START_TOKEN_ID
            length, cnt1, cnt2 = 0, 0, 0
            for index, conv in enumerate(candidate_data):
                pos = 1
                flag = True
                for turn in conv:
                    for token in turn:
                        if pos == max_length:
                            flag = False
                            break
                        engage_input_ids[index][pos] = token
                        targets[index][pos - 1] = token
                        pos += 1
                    if pos == max_length:
                        break
                engage_cls_indices[index] = pos - 1
                engage_attention_mask[index][:pos] = True
                length += pos
                if flag:
                    cnt1 += 1
                else:
                    cnt2 += 1
            with open(engage_input_ids_file, 'wb') as f:
                pickle.dump(engage_input_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(engage_cls_indices_file, 'wb') as f:
                pickle.dump(engage_cls_indices, f)
            with open(engage_attention_mask_file, 'wb') as f:
                pickle.dump(engage_attention_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(targets_file, 'wb') as f:
                pickle.dump(targets, f, protocol=pickle.HIGHEST_PROTOCOL)
            assert ((targets != PAD_TOKEN_ID) & (targets != IGNORE_TOKEN_ID)).any(axis=1).all()
            print(prefix, mode, length / num, cnt1, cnt2)


def debug(encoder_max_length, decoder_max_length):
    debug_index = 256
    with open('engagerec.debug', 'a', encoding='utf-8') as debug_f:
        for mode in ['dev', 'test']:
            for prefix in ['funny', 'technology', 'todayilearned']:
                history_input_ids_file = os.path.join(reddit_engage_dataset_roots[prefix], 'history_input_ids-%s-%d.pkl' % (mode, encoder_max_length))
                engage_input_ids_file = os.path.join(reddit_engage_dataset_roots[prefix], 'engage_input_ids-%s-%d.pkl' % (mode, decoder_max_length))
                truth_file = os.path.join(reddit_engage_dataset_roots[prefix], 'truth-%s.txt' % mode)
                with open(history_input_ids_file, 'rb') as f:
                    history_input_ids = pickle.load(f)[debug_index]
                with open(engage_input_ids_file, 'rb') as f:
                    engage_input_ids = pickle.load(f)
                with open(truth_file, 'r', encoding='utf-8') as f:
                    labels = []
                    for line in f:
                        item = line.strip()
                        if len(item) > 0:
                            labels.append(json.loads(item.split(' ')[1]))
                debug_f.write('\n%s\t%s\n' % (mode, prefix))
                pos = encoder_max_length
                for i in range(encoder_max_length):
                    if history_input_ids[i] == PAD_TOKEN_ID:
                        pos = i
                        break
                debug_f.write('History\n%s\nCandidates\n' % tokenizer.decode(history_input_ids[:pos]))
                for i, label in enumerate(labels[debug_index]):
                    if label == 1:
                        engage_input_ids_ = engage_input_ids[i]
                        pos = decoder_max_length
                        for j in range(decoder_max_length):
                            if engage_input_ids_[j] == PAD_TOKEN_ID:
                                pos = j
                                break
                        debug_f.write('%d\t%s\n' % (i, tokenizer.decode(engage_input_ids_[:pos])))


if __name__ == '__main__':
    encoder_max_length = 1024
    decoder_max_length = 576
    download_engageDatasets()
    prepreprocess()
    debug_prepreprocess()
    preprocess_history_inputs(encoder_max_length)
    preprocess_candidate_inputs(decoder_max_length)
    debug(encoder_max_length, decoder_max_length)
