import sys
sys.path.append('..')
import os
import zipfile
import shutil
from tqdm import tqdm
import json
import pickle
import numpy as np
from transformers.models.bart import BartTokenizer
import nltk
BOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
DECODER_START_TOKEN_ID = 2 # a unique hack of BART, also refer to https://github.com/huggingface/transformers/issues/5212
IGNORE_TOKEN_ID = -100
MAX_HISTORY_NUM = 255
MAX_IMPRESSION_NUM = 300
newsrec_root = 'newsRec'
if not os.path.exists(newsrec_root):
    os.mkdir(newsrec_root)
MIND_small_dataset_root = os.path.join(newsrec_root, 'MIND-small')
MIND_large_dataset_root = os.path.join(newsrec_root, 'MIND-large')
dataset_roots = [MIND_small_dataset_root, MIND_large_dataset_root]
if not os.path.exists(MIND_small_dataset_root):
    os.mkdir(MIND_small_dataset_root)
if not os.path.exists(MIND_large_dataset_root):
    os.mkdir(MIND_large_dataset_root)
if not os.path.exists('../../download'):
    os.mkdir('../../download')
if not os.path.exists('../../backbone_models'):
    os.mkdir('../../backbone_models')
if not os.path.exists('../../backbone_models/bart-base'):
    os.system('git lfs install')
    os.system('git clone https://huggingface.co/facebook/bart-base ../../backbone_models/bart-base')
tokenizer = BartTokenizer.from_pretrained('../../backbone_models/bart-base')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
stopwords1 = [word for word in stopwords] + [word[0].upper() + word[1:] for word in stopwords] + [word.upper() for word in stopwords]
stopwords2 = [' ' + word for word in stopwords1]
stopword_tokens1 = [tokenizer(word, add_special_tokens=False)['input_ids'] for word in stopwords1]
stopword_tokens2 = [tokenizer(word, add_special_tokens=False)['input_ids'] for word in stopwords2]
with open('../../backbone_models/bart-base/vocab.json', 'r', encoding='utf-8') as f:
    tokenizer_vocab = {v: k for k, v in json.load(f).items()}
IGNORE_TARGET_STOP_WORDS = True # stop words always get low perplexity in languages regardless the context, this is to flag whether ignore target stop words


def download_MIND():
    # since MIND-small does not have test set, we follow HieRec (https://github.com/taoqi98/HieRec) to split part of data from the training set as dev set, and dev set serves as test set
    if MIND_small_dataset_root in dataset_roots and not all(map(lambda x: os.path.exists(os.path.join(MIND_small_dataset_root, x)), ['train/news.tsv', 'train/behaviors.tsv', 'dev/news.tsv', 'dev/behaviors.tsv', 'test/news.tsv', 'test/behaviors.tsv'])):
        for mode in ['train+dev', 'test']:
            if os.path.exists(os.path.join(MIND_small_dataset_root, mode)):
                shutil.rmtree(os.path.join(MIND_small_dataset_root, mode))
        train_dev_zip = '../../download/MINDsmall_train.zip'
        test_zip = '../../download/MINDsmall_dev.zip'
        if not os.path.exists(train_dev_zip):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip -P ../../download')
        if not os.path.exists(test_zip):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip -P ../../download')
        with zipfile.ZipFile(train_dev_zip) as f:
            f.extractall(os.path.join(MIND_small_dataset_root, 'train+dev'))
        if not os.path.exists(os.path.join(MIND_small_dataset_root, 'train')):
            os.mkdir(os.path.join(MIND_small_dataset_root, 'train'))
        shutil.copy(os.path.join(MIND_small_dataset_root, 'train+dev', 'news.tsv'), os.path.join(MIND_small_dataset_root, 'train', 'news.tsv'))
        if not os.path.exists(os.path.join(MIND_small_dataset_root, 'dev')):
            os.mkdir(os.path.join(MIND_small_dataset_root, 'dev'))
        shutil.copy(os.path.join(MIND_small_dataset_root, 'train+dev', 'news.tsv'), os.path.join(MIND_small_dataset_root, 'dev', 'news.tsv'))
        with open(os.path.join(MIND_small_dataset_root, 'train+dev', 'behaviors.tsv'), 'r', encoding='utf-8') as f:
            with open(os.path.join(MIND_small_dataset_root, 'train', 'behaviors.tsv'), 'w', encoding='utf-8') as f1, open(os.path.join(MIND_small_dataset_root, 'dev', 'behaviors.tsv'), 'w', encoding='utf-8') as f2:
                cnt1, cnt2 = 0, 0
                for index, line in enumerate(f):
                    items = line.split('\t')
                    if index % 100 <= 95 or cnt2 >= 2500: # although we do not think MIND-small is a good dataset, we have to split datasets benchmarking previous work (HieRec, https://github.com/taoqi98/HieRec). We think 2.5k dev set is enough to early-stop training.
                        cnt1 += 1
                        f1.write('\t'.join([str(cnt1)] + items[1:]))
                    else:
                        cnt2 += 1
                        f2.write('\t'.join([str(cnt2)] + items[1:]))
        with zipfile.ZipFile(test_zip) as f:
            f.extractall(os.path.join(MIND_small_dataset_root, 'test'))
    if MIND_large_dataset_root in dataset_roots and not all(map(lambda x: os.path.exists(os.path.join(MIND_large_dataset_root, x)), ['train/news.tsv', 'train/behaviors.tsv', 'dev/news.tsv', 'dev/behaviors.tsv', 'test/news.tsv', 'test/behaviors.tsv'])):
        for mode in ['train', 'dev', 'test']:
            if os.path.exists(os.path.join(MIND_large_dataset_root, mode)):
                shutil.rmtree(os.path.join(MIND_large_dataset_root, mode))
        train_zip = '../../download/MINDlarge_train.zip'
        dev_zip = '../../download/MINDlarge_dev.zip'
        test_zip = '../../download/MINDlarge_test.zip'
        if not os.path.exists(train_zip):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip -P ../../download')
        if not os.path.exists(dev_zip):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip -P ../../download')
        if not os.path.exists(test_zip):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip -P ../../download')
        with zipfile.ZipFile(train_zip) as f:
            f.extractall(os.path.join(MIND_large_dataset_root, 'train'))
        with zipfile.ZipFile(dev_zip) as f:
            f.extractall(os.path.join(MIND_large_dataset_root, 'dev'))
        with zipfile.ZipFile(test_zip) as f:
            f.extractall(os.path.join(MIND_large_dataset_root, 'test'))


def replace_stopword_tokens(tokens: np.array):
    assert len(tokens.shape) == 1
    N = tokens.shape[0]
    for stopword_tokens in stopword_tokens1:
        M = len(stopword_tokens)
        for i in range(M):
            if stopword_tokens[i] != tokens[i]:
                break
        else:
            if tokens[M] != IGNORE_TOKEN_ID and tokenizer_vocab[tokens[M]][0] == 'Ġ':
                tokens[:M] = IGNORE_TOKEN_ID
                break
    for i in range(1, N):
        if tokens[i] != IGNORE_TOKEN_ID and tokens[i] != PAD_TOKEN_ID:
            for stopword_tokens in stopword_tokens2:
                M = len(stopword_tokens)
                if i + M < N:
                    for j in range(M):
                        if stopword_tokens[j] != tokens[i + j]:
                            break
                    else:
                        if tokens[i + M] != IGNORE_TOKEN_ID and tokenizer_vocab[tokens[i + M]][0] == 'Ġ':
                            tokens[i: i + M] = IGNORE_TOKEN_ID
                            break
    assert np.any((tokens != IGNORE_TOKEN_ID) & (tokens != PAD_TOKEN_ID))
    return tokens


def tokenize_MIND_news():
    news_max_length = 0
    for dataset_root in dataset_roots:
        length = 0
        MIND_news_tokens = {}
        for mode in ['train', 'dev', 'test']:
            with open(os.path.join(dataset_root, mode, 'news.tsv'), 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line) > 0:
                        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                        if news_ID not in MIND_news_tokens:
                            while True:
                                _title = title
                                title = title.strip()
                                title = title.strip('.!,;')
                                while '  ' in title:
                                    title = title.replace('  ', ' ')
                                if title == _title:
                                    break
                            news_input_ids = tokenizer(title, add_special_tokens=False)['input_ids'] + [EOS_TOKEN_ID]
                            MIND_news_tokens[news_ID] = news_input_ids
                            _length = len(news_input_ids)
                            length += _length
                            news_max_length = max(_length, news_max_length)
        with open(os.path.join(dataset_root, 'news.pkl'), 'wb') as f:
            pickle.dump(MIND_news_tokens, f)
        with open(os.path.join(dataset_root, 'mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({news_ID: i for i, news_ID in enumerate(MIND_news_tokens)}, f)
        print(dataset_root, 'average news tokens :', length / len(MIND_news_tokens))
    print('News max length :', news_max_length + 2)


def preprocess_MIND_history_tokens(max_length):
    max_history_length = 0
    for MIND_dataset_root in dataset_roots:
        with open(os.path.join(MIND_dataset_root, 'news.pkl'), 'rb') as f:
            news_tokens = pickle.load(f)
        for mode in ['train', 'dev', 'test']:
            num = 0
            with open(os.path.join(MIND_dataset_root, mode, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        num += 1
            print('Preprocessing %s %s : %d' % (MIND_dataset_root, mode, num))
            history_input_ids = np.full([num, max_length], PAD_TOKEN_ID, dtype=np.uint16)
            history_segment_ids = np.full([num, max_length], MAX_HISTORY_NUM, dtype=np.uint8)
            history_global_attention_mask = np.zeros([num, max_length], dtype=bool)
            history_local_position_ids = np.zeros([num, max_length], dtype=np.int16)
            history_input_ids[:, 0] = BOS_TOKEN_ID
            history_segment_ids[:, 0] = 0
            history_global_attention_mask[:, 0] = True
            with open(os.path.join(MIND_dataset_root, mode, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
                for index, line in tqdm(enumerate(f)):
                    if len(line.strip()) > 0:
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        if len(history) != 0:
                            history = history.strip().split(' ')
                            history.reverse()
                            pos = 1
                            for history_index, history_news in enumerate(history):
                                history_tokens = news_tokens[history_news]
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
                                    break
                            max_history_length = max(max_history_length, pos)
            with open(os.path.join(MIND_dataset_root, 'history_input_ids-%s-%d.pkl' % (mode, max_length)), 'wb') as f:
                pickle.dump(history_input_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(MIND_dataset_root, 'history_segment_ids-%s-%d.pkl' % (mode, max_length)), 'wb') as f:
                pickle.dump(history_segment_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(MIND_dataset_root, 'history_global_attention_mask-%s-%d.pkl' % (mode, max_length)), 'wb') as f:
                pickle.dump(history_global_attention_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(MIND_dataset_root, 'history_local_position_ids-%s-%d.pkl' % (mode, max_length)), 'wb') as f:
                pickle.dump(history_local_position_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('History max length :', max_history_length)


def preprocess_MIND_news_tokens(max_length):
    for MIND_dataset_root in dataset_roots:
        target_token_num = 0
        with open(os.path.join(MIND_dataset_root, 'mapping.json'), 'r', encoding='utf-8') as f:
            news_mapping = json.load(f)
        with open(os.path.join(MIND_dataset_root, 'news.pkl'), 'rb') as f:
            news_tokens = pickle.load(f)
        num = len(news_mapping)
        assert len(news_tokens) == num
        news_input_ids = np.full([num, max_length], PAD_TOKEN_ID, dtype=np.int32)
        news_cls_indices = np.zeros([num], dtype=np.int32)
        news_attention_mask = np.zeros([num, max_length], dtype=bool)
        targets = np.full([num, max_length], PAD_TOKEN_ID, dtype=np.int64)
        news_input_ids[:, 0] = DECODER_START_TOKEN_ID
        for news_ID, input_ids in news_tokens.items():
            index = news_mapping[news_ID]
            token_num = len(input_ids)
            assert token_num + 1 <= max_length
            for i in range(token_num):
                news_input_ids[index][i + 1] = input_ids[i]
                targets[index][i] = input_ids[i]
                target_token_num += 1
            news_cls_indices[index] = token_num
            news_attention_mask[index][:token_num + 1] = True
            assert targets[index][token_num - 1] == EOS_TOKEN_ID
            targets[index][token_num - 1] = IGNORE_TOKEN_ID
            if IGNORE_TARGET_STOP_WORDS:
                targets[index] = replace_stopword_tokens(targets[index])
        with open(os.path.join(MIND_dataset_root, 'news_input_ids-%d.pkl' % max_length), 'wb') as f:
            pickle.dump(news_input_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(MIND_dataset_root, 'news_cls_indices-%d.pkl' % max_length), 'wb') as f:
            pickle.dump(news_cls_indices, f)
        with open(os.path.join(MIND_dataset_root, 'news_attention_mask-%d.pkl' % max_length), 'wb') as f:
            pickle.dump(news_attention_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(MIND_dataset_root, 'targets-%d.pkl' % max_length), 'wb') as f:
            pickle.dump(targets, f, protocol=pickle.HIGHEST_PROTOCOL)
        assert ((targets != PAD_TOKEN_ID) & (targets != IGNORE_TOKEN_ID)).any(axis=1).all()
        print(MIND_dataset_root, ': target_token_num =', target_token_num / num)


def MIND_indexing():
    for MIND_dataset_root in dataset_roots:
        with open(os.path.join(MIND_dataset_root, 'mapping.json'), 'r', encoding='utf-8') as f:
            news_mapping = json.load(f)
        for mode in ['train', 'dev', 'test']:
            indexing = []
            if mode == 'train':
                with open(os.path.join(MIND_dataset_root, 'train/behaviors.tsv'), 'r', encoding='utf-8') as f:
                    for history_index, line in enumerate(f):
                        if len(line.strip()) > 0:
                            impression_ID, user_ID, time, history, impressions = line.split('\t')
                            positive_samples = []
                            negative_samples = []
                            for impression in impressions.strip().split(' '):
                                if impression[-2:] == '-1':
                                    positive_samples.append(news_mapping[impression[:-2]])
                                elif impression[-2:] == '-0':
                                    negative_samples.append(news_mapping[impression[:-2]])
                                else:
                                    raise Exception('Impression error : ' + impression)
                            for positive_sample in positive_samples:
                                assert positive_sample not in negative_samples
                            indexing.append([history_index, positive_samples, negative_samples])
                with open(os.path.join(MIND_dataset_root, 'indexing-train.pkl'), 'wb') as f:
                    pickle.dump(indexing, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                flag = not (MIND_dataset_root == MIND_large_dataset_root and mode == 'test') # labels of MIND-large test set is unrevealed
                if flag:
                    truth_f = open(os.path.join(MIND_dataset_root, mode, 'truth.txt'), 'w', encoding='utf-8')
                with open(os.path.join(MIND_dataset_root, mode, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
                    for history_index, line in enumerate(f):
                        if len(line.strip()) > 0:
                            impression_ID, user_ID, time, history, impressions = line.split('\t')
                            index = int(impression_ID)
                            impression_indices = []
                            if flag:
                                labels = []
                                for impression in impressions.strip().split(' '):
                                    impression_indices.append(news_mapping[impression[:-2]])
                                    if impression[-2:] == '-1':
                                        labels.append(1)
                                    elif impression[-2:] == '-0':
                                        labels.append(0)
                                    else:
                                        raise Exception('Impression error : ' + impression)
                                truth_f.write(('' if index == 1 else '\n') + str(index) + ' ' + str(labels).replace(' ', ''))
                            else:
                                for impression in impressions.strip().split(' '):
                                    impression_indices.append(news_mapping[impression])
                            assert len(impression_indices) <= MAX_IMPRESSION_NUM
                            indexing.append([history_index, impression_indices])
                if flag:
                    truth_f.close()
                with open(os.path.join(MIND_dataset_root, 'indexing-' + mode + '.pkl'), 'wb') as f:
                    pickle.dump(indexing, f, protocol=pickle.HIGHEST_PROTOCOL)


def debug(encoder_max_length, decoder_max_length):
    debug_index = 1024
    with open('newsrec.debug', 'w', encoding='utf-8') as debug_f:
        for dataset in map(os.path.basename, dataset_roots):
            dataset_root = 'newsRec/' + dataset
            for mode in ['train', 'dev', 'test']:
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
                debug_f.write('\n')
            with open(os.path.join(dataset_root, 'news_input_ids-%d.pkl' % decoder_max_length), 'rb') as f:
                news_input_ids = pickle.load(f)[debug_index]
            with open(os.path.join(dataset_root, 'news_cls_indices-%d.pkl' % decoder_max_length), 'rb') as f:
                news_cls_index = pickle.load(f)[debug_index]
            with open(os.path.join(dataset_root, 'news_attention_mask-%d.pkl' % decoder_max_length), 'rb') as f:
                news_attention_mask = pickle.load(f)[debug_index]
            debug_f.write('\n' + dataset + '\tnews :\n')
            assert news_input_ids[news_cls_index] == EOS_TOKEN_ID
            input_ids = []
            for i in range(decoder_max_length):
                if news_attention_mask[i]:
                    input_ids.append(news_input_ids[i])
            text = tokenizer.decode(input_ids)
            debug_f.write(text + '\n\n')


if __name__ == '__main__':
    encoder_max_length = 1024
    decoder_max_length = 64
    download_MIND()
    tokenize_MIND_news()
    preprocess_MIND_history_tokens(encoder_max_length)
    preprocess_MIND_news_tokens(decoder_max_length)
    MIND_indexing()
    debug(encoder_max_length, decoder_max_length)
