import os
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.cuda import amp
from transformers.models.UniTRec import UniTRecConfig, UniTRecModel
from textRec_datasets.quoterec_dataset import QuoteRecTrainDataset, QuoteRecValDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import argparse
import json
from colorama import Fore
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils.evaluate import scoring
from utils.misc import AvgMetric, AverageRanking, write_predictions
import gc
import shutil


def parse_config():
    parser = argparse.ArgumentParser(description='QuoteRec')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode')
    parser.add_argument('--backbone_model', type=str, default='../backbone_models/bart-base/', choices=['../backbone_models/bart-base/', '../backbone_models/bart-large/'], help='Backbone BART model')
    parser.add_argument('--init_temperature', type=float, default=1, help='Initial temperature')
    parser.add_argument('--max_temperature', type=float, default=100, help='Max temperature')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epoch', type=int, default=5, help='Epoch')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--gradient_clip_norm', type=float, default=1, help='Gradient clip norm')
    parser.add_argument('--lr_warm_up_ratio', type=float, default=0.05, help='Larning rate warm-up ratio')
    parser.add_argument('--fp16', type=int, default=1, choices=[0, 1], help='Whether use fp16')
    parser.add_argument('--gradient_checkpoint', type=int, default=1, choices=[0, 1], help='Whether use gradient checkpointing')
    parser.add_argument('--negative_sample_num', type=int, default=19, help='Number of negative samples') # This configuration empirically follows https://aclanthology.org/2022.acl-long.27
    parser.add_argument('--ppl_loss', type=int, default=1, choices=[0, 1], help='Whether use perplexity contrastive loss')
    parser.add_argument('--dis_loss', type=int, default=1, choices=[0, 1], help='Whether use discriminative contrastive loss')
    parser.add_argument('--task', type=str, default='quoteRec/Reddit-quote', choices=['quoteRec/Reddit-quote', 'quoteRec/QuoteR'], help='Text-based recommendation tasks')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--val_interval', type=int, default=10000, help='Validation interval')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--test_model_path', type=str, default='', help='Test model path')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank')
    parser.add_argument('--encoder_local_attention_layers', type=int, default=3, choices=[i for i in range(7)], help='Experimental: encoder local attention layers')
    parser.add_argument('--average_ranking', type=str, default='normalized', choices=['ordinal', 'normalized', 'harmonic'], help='Ranking method to average perplexity and discriminative scores')
    args = parser.parse_args()
    if args.task == 'quoteRec/Reddit-quote':
        args.encoder_seq_len = 1024
        args.decoder_seq_len = 72
    elif args.task == 'quoteRec/QuoteR':
        args.encoder_seq_len = 384
        args.decoder_seq_len = 84
    else:
        raise Exception('Unexpected quote task : ' + args.task + ' (should be chosen from [\'quoteRec/Reddit-quote\', \'quoteRec/QuoteR\'])')
    assert os.path.exists(args.backbone_model) or args.mode != 'train', 'Backbone BART does not exist at ' + args.backbone_model
    args.ppl_loss = bool(args.ppl_loss)
    args.dis_loss = bool(args.dis_loss)
    AverageRanking.set_average_ranking_method(args.average_ranking)
    args.timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if args.mode == 'train':
        args.fp16 = bool(args.fp16)
        args.gradient_checkpoint = bool(args.gradient_checkpoint)
        args.ppl_eval = args.ppl_loss
        args.dis_eval = args.dis_loss
        if args.local_rank in [-1, 0]:
            args.log_dir = os.path.join('logs', args.task, args.timestamp)
            if not os.path.exists(args.log_dir):
                os.makedirs(args.log_dir)
            args.model_dir = os.path.join('ckpt_models', args.task, args.timestamp)
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)
            args.prediction_dir = os.path.join('predictions', args.task, args.timestamp)
            if not os.path.exists(args.prediction_dir):
                os.makedirs(args.prediction_dir)
            args.best_model_dir = os.path.join('best_model', args.task, args.timestamp)
            if not os.path.exists(args.best_model_dir):
                os.makedirs(args.best_model_dir)
    else:
        assert os.path.exists(args.test_model_path), 'Test model not exists: ' + args.test_model_path
        with open(os.path.join(args.test_model_path, 'args.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
        args.ppl_eval = config['ppl_loss']
        args.dis_eval = config['dis_loss']
    assert args.ppl_loss or args.dis_loss, 'At least one type of [ppl_loss, dis_loss] must be specified'
    assert args.ppl_eval or args.dis_eval, 'At least one type of [ppl_eval, dis_eval] must be specified'
    assert torch.cuda.is_available()
    if args.local_rank == -1:
        torch.cuda.set_device(args.device_id)
    else:
        torch.cuda.set_device(torch.device('cuda:{}'.format(args.local_rank)))
        os.environ['MASTER_ADDR'] = 'localhost'
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, 14400))
        if args.local_rank == 0:
            for i in range(1, dist.get_world_size()):
                with open('%s-%d.tmp' % (os.environ['MASTER_PORT'], i), 'w', encoding='utf-8') as f:
                    f.write(args.timestamp)
        dist.barrier()
        if args.local_rank > 0:
            tmp_file = '%s-%d.tmp' % (os.environ['MASTER_PORT'], args.local_rank)
            with open(tmp_file, 'r', encoding='utf-8') as f:
                args.timestamp = f.read().strip()
            os.remove(tmp_file)
            if args.mode == 'train':
                args.prediction_dir = os.path.join('predictions', args.task, args.timestamp)
                args.best_model_dir = os.path.join('best_model', args.task, args.timestamp)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    if args.local_rank in [-1, 0]:
        attribute_dict = dict(vars(args))
        print(Fore.RED + '*' * 32 + ' UniTRec ' + '*' * 32)
        for attribute in attribute_dict:
            print(attribute + ' : ' + str(attribute_dict[attribute]))
        print('*' * 32 + ' UniTRec ' + '*' * 32 + Fore.RESET)
    return args


def build_UniTRec_model(args):
    config = UniTRecConfig.from_pretrained(args.backbone_model)
    config.gradient_checkpoint = args.gradient_checkpoint
    config.encoder_seq_len = args.encoder_seq_len
    config.decoder_seq_len = args.decoder_seq_len
    config.dropout = 0
    config.activation_dropout = 0
    config.attention_dropout = 0
    config.init_temperature = args.init_temperature
    config.max_temperature = args.max_temperature
    config.encoder_local_attention_layers = args.encoder_local_attention_layers # Experimental
    UniTRec = UniTRecModel(config=config, dis_scoring=args.dis_loss, ppl_scoring=args.ppl_loss)
    UniTRec.load_bart(args.backbone_model)
    if args.gradient_checkpoint:
        UniTRec.gradient_checkpointing_enable()
    return config, UniTRec.cuda()


def detect_invalid_gradient(model):
    for p in model.parameters():
        if p.requires_grad and (torch.isinf(p.grad).any() or torch.isnan(p.grad).any()):
            model.zero_grad()
            return


def train(args):
    config, model = build_UniTRec_model(args)
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    train_dataset = QuoteRecTrainDataset(args)
    train_dataset.negative_sampling(epoch=1)
    if args.local_rank == -1:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = QuoteRecValDataset(args, mode='dev')
        test_dataset = QuoteRecValDataset(args, mode='test')
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_dataset = QuoteRecValDataset(args, mode='dev', rank=args.local_rank, world_size=dist.get_world_size())
        test_dataset = QuoteRecValDataset(args, mode='test', rank=args.local_rank, world_size=dist.get_world_size())
    assert (model.module.IGNORE_TOKEN_ID == train_dataset.IGNORE_TOKEN_ID if hasattr(model, 'module') else model.IGNORE_TOKEN_ID == train_dataset.IGNORE_TOKEN_ID)
    no_decay = ['bias', 'layer_norm.weight', 'layernorm_embedding.weight', 'fc.weight', 'temperature']
    for n, p in model.named_parameters():
        if 'bias' in n.lower() or 'norm' in n.lower() or len(p.squeeze().shape) == 1:
            assert any(nd in n.lower() for nd in no_decay), 'Parameter decay error : ' + n
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay) and p.requires_grad], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-7, betas=(0.9, 0.98))
    num_training_steps = len(train_dataloader) * args.epoch
    num_warmup_steps = int(num_training_steps * args.lr_warm_up_ratio)
    lr_scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    if args.local_rank in [-1, 0]:
        print('Training steps :', num_training_steps)
        writer = SummaryWriter(log_dir=args.log_dir, filename_suffix='.log')
    if args.fp16:
        scaler = amp.GradScaler()
    iteration, iteration_ppl_loss, iteration_dis_loss, iteration_loss = 0, 0, 0, 0
    best_val_result = AvgMetric(0, 0, 0, 0, 0, 0, 0)
    best_val_epoch = 0

    for epoch in tqdm(range(1, args.epoch + 1)) if args.local_rank in [-1, 0] else range(1, args.epoch + 1):
        model.train()
        train_dataset.negative_sampling(epoch=epoch)
        if args.local_rank == -1:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_sampler.set_epoch(epoch)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        epoch_ppl_loss, epoch_dis_loss, epoch_loss = 0, 0, 0
        for history_input_ids, history_segment_ids, history_global_attention_mask, history_local_position_ids, candidate_input_ids, candidate_cls_indices, targets in train_dataloader:
            history_input_ids = history_input_ids.cuda(non_blocking=True)
            history_segment_ids = history_segment_ids.cuda(non_blocking=True)
            history_global_attention_mask = history_global_attention_mask.cuda(non_blocking=True)
            history_local_position_ids = history_local_position_ids.cuda(non_blocking=True)
            candidate_input_ids = candidate_input_ids.cuda(non_blocking=True)
            candidate_cls_indices = candidate_cls_indices.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            if args.fp16:
                with amp.autocast():
                    ppl_scores, dis_scores = model(history_input_ids, history_segment_ids, history_global_attention_mask, history_local_position_ids, candidate_input_ids, candidate_cls_indices, targets)
                    if args.ppl_loss:
                        ppl_loss = -log_softmax(ppl_scores, dim=1).select(dim=1, index=0).mean()
                    if args.dis_loss:
                        dis_loss = -log_softmax(dis_scores, dim=1).select(dim=1, index=0).mean()
                    if args.ppl_loss and not args.dis_loss:
                        loss = ppl_loss
                    elif not args.ppl_loss and args.dis_loss:
                        loss = dis_loss
                    else:
                        loss = ppl_loss + dis_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                detect_invalid_gradient(model)
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                ppl_scores, dis_scores = model(history_input_ids, history_segment_ids, history_global_attention_mask, history_local_position_ids, candidate_input_ids, candidate_cls_indices, targets)
                if args.ppl_loss:
                    ppl_loss = -log_softmax(ppl_scores, dim=1).select(dim=1, index=0).mean()
                if args.dis_loss:
                    dis_loss = -log_softmax(dis_scores, dim=1).select(dim=1, index=0).mean()
                if args.ppl_loss and not args.dis_loss:
                    loss = ppl_loss
                elif not args.ppl_loss and args.dis_loss:
                    loss = dis_loss
                else:
                    loss = ppl_loss + dis_loss
                loss.backward()
                detect_invalid_gradient(model)
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            _ppl_loss = ppl_loss.item() if args.ppl_loss else 0
            _dis_loss = dis_loss.item() if args.dis_loss else 0
            _loss = loss.item()
            epoch_ppl_loss += _ppl_loss
            epoch_dis_loss += _dis_loss
            epoch_loss += _loss
            iteration_ppl_loss += _ppl_loss
            iteration_dis_loss += _dis_loss
            iteration_loss += _loss
            iteration += 1
            if iteration % args.log_interval == 0:
                iteration_loss /= args.log_interval
                iteration_ppl_loss /= args.log_interval
                iteration_dis_loss /= args.log_interval
                if args.local_rank in [-1, 0]:
                    temperature = model.module.temperature.item() if hasattr(model, 'module') else model.temperature.item()
                    if args.ppl_loss and not args.dis_loss:
                        print('Iteration : %d\t\tLR = %.6f\t\tTemperature = %.4f\t\tLoss = %.2f\t\tppl_loss = %.2f' % (iteration, lr_scheduler.get_last_lr()[0], temperature, iteration_loss, iteration_ppl_loss))
                    elif not args.ppl_loss and args.dis_loss:
                        print('Iteration : %d\t\tLR = %.6f\t\tTemperature = %.4f\t\tLoss = %.2f\t\tdis_loss = %.2f' % (iteration, lr_scheduler.get_last_lr()[0], temperature, iteration_loss, iteration_dis_loss))
                    else:
                        print('Iteration : %d\t\tLR = %.6f\t\tTemperature = %.4f\t\tLoss = %.2f\t\tppl_loss = %.2f\t\tdis_loss = %.2f' % (iteration, lr_scheduler.get_last_lr()[0], temperature, iteration_loss, iteration_ppl_loss, iteration_dis_loss))
                    writer.add_scalar('Iteration Loss', iteration_loss, iteration)
                    if args.ppl_loss:
                        writer.add_scalar('Iteration ppl_loss', iteration_ppl_loss, iteration)
                    if args.dis_loss:
                        writer.add_scalar('Iteration dis_loss', iteration_dis_loss, iteration)
                    writer.add_scalar('Iteration Temperature', temperature, iteration)
                iteration_ppl_loss, iteration_dis_loss, iteration_loss = 0, 0, 0
            if iteration % args.val_interval == 0:
                val_model = model.module if hasattr(model, 'module') else model
                result_file = os.path.join(args.prediction_dir, 'iteration-%d.txt' % iteration)
                if args.local_rank == -1:
                    auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = inference(args, val_model, val_dataset, result_file, return_scores=True)
                else: # distributed inference and aggregation results
                    inference(args, val_model, val_dataset, result_file + '-' + str(args.local_rank), return_scores=False)
                    dist.barrier()
                    if args.local_rank == 0:
                        with open(result_file, 'w', encoding='utf-8') as f:
                            index = 0
                            for i in range(dist.get_world_size()):
                                with open(result_file + '-' + str(i), 'r', encoding='utf-8') as f_:
                                    for line in f_:
                                        if len(line.strip()) > 0:
                                            index += 1
                                            result_line = ('' if index == 1 else '\n') + str(index) + line[line.find(' '):].strip('\n')
                                            f.write(result_line)
                        with open(val_dataset.truth_file, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
                            auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = scoring(truth_f, result_f, onehot=True)
                            auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = auc * 100, mrr * 100, ndcg5 * 100, ndcg10 * 100, hr1 * 100, hr5 * 100, hr10 * 100 # return percentage scores
                if args.local_rank in [-1, 0]:
                    print('Validation iteration : %d\nAUC = %.2f\nMRR = %.2f\nnDCG@5 = %.2f\nnDCG@10 = %.2f\nHR@1 = %.2f\nHR@5 = %.2f\nHR@10 = %.2f' % (iteration, auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10))
                    writer.add_scalar('Iteration AUC', auc, iteration)
                    writer.add_scalar('Iteration MRR', mrr, iteration)
                    writer.add_scalar('Iteration nDCG@5', ndcg5, iteration)
                    writer.add_scalar('Iteration nDCG@10', ndcg10, iteration)
                    writer.add_scalar('Iteration HR@1', hr1, iteration)
                    writer.add_scalar('Iteration HR@5', hr5, iteration)
                    writer.add_scalar('Iteration HR@10', hr10, iteration)
                    val_model.save_pretrained(os.path.join(args.model_dir, 'iteration-' + str(iteration)))
                    config.save_pretrained(os.path.join(args.model_dir, 'iteration-' + str(iteration)))
                    with open(os.path.join(args.model_dir, 'iteration-' + str(iteration), 'args.json'), 'w', encoding='utf-8') as f:
                        json.dump(dict(vars(args)), f)
                gc.collect()
                torch.cuda.empty_cache()
                model.train()

        epoch_loss /= len(train_dataloader)
        epoch_ppl_loss /= len(train_dataloader)
        epoch_dis_loss /= len(train_dataloader)
        val_model = model.module if hasattr(model, 'module') else model
        result_file = os.path.join(args.prediction_dir, 'epoch-%d.txt' % epoch)
        if args.local_rank == -1:
            auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = inference(args, val_model, val_dataset, result_file, return_scores=True)
        else: # distributed inference and aggregation results
            inference(args, val_model, val_dataset, result_file + '-' + str(args.local_rank), return_scores=False)
            dist.barrier()
            if args.local_rank == 0:
                with open(result_file, 'w', encoding='utf-8') as f:
                    index = 0
                    for i in range(dist.get_world_size()):
                        with open(result_file + '-' + str(i), 'r', encoding='utf-8') as f_:
                            for line in f_:
                                if len(line.strip()) > 0:
                                    index += 1
                                    result_line = ('' if index == 1 else '\n') + str(index) + line[line.find(' '):].strip('\n')
                                    f.write(result_line)
                with open(val_dataset.truth_file, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
                    auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = scoring(truth_f, result_f, onehot=True)
                    auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = auc * 100, mrr * 100, ndcg5 * 100, ndcg10 * 100, hr1 * 100, hr5 * 100, hr10 * 100 # return percentage scores
        if args.local_rank in [-1, 0]:
            if args.ppl_loss and not args.dis_loss:
                print('Epoch : %d\t\tLoss = %.2f\t\tppl_loss = %.2f' % (epoch, epoch_loss, epoch_ppl_loss))
            elif not args.ppl_loss and args.dis_loss:
                print('Epoch : %d\t\tLoss = %.2f\t\tdis_loss = %.2f' % (epoch, epoch_loss, epoch_dis_loss))
            else:
                print('Epoch : %d\t\tLoss = %.2f\t\tppl_loss = %.2f\t\tdis_loss = %.2f' % (epoch, epoch_loss, epoch_ppl_loss, epoch_dis_loss))
            writer.add_scalar('Epoch Loss', epoch_loss, epoch)
            if args.ppl_loss:
                writer.add_scalar('Epoch ppl_loss', epoch_ppl_loss, epoch)
            if args.dis_loss:
                writer.add_scalar('Epoch dis_loss', epoch_dis_loss, epoch)
            val_result = AvgMetric(auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10)
            if val_result > best_val_result:
                best_val_result = val_result
                best_val_epoch = epoch
            print('Validation epoch : %d\nAUC = %.2f\nMRR = %.2f\nnDCG@5 = %.2f\nnDCG@10 = %.2f\nHR@1 = %.2f\nHR@5 = %.2f\nHR@10 = %.2f' % (epoch, auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10))
            print(Fore.BLUE + ('Best epoch : %d\nBest result = %s' % (best_val_epoch, str(best_val_result))) + Fore.RESET)
            writer.add_scalar('Epoch AUC', auc, epoch)
            writer.add_scalar('Epoch MRR', mrr, epoch)
            writer.add_scalar('Epoch nDCG@5', ndcg5, epoch)
            writer.add_scalar('Epoch nDCG@10', ndcg10, epoch)
            writer.add_scalar('Epoch HR@1', hr1, epoch)
            writer.add_scalar('Epoch HR@5', hr5, epoch)
            writer.add_scalar('Epoch HR@10', hr10, epoch)
            val_model.save_pretrained(os.path.join(args.model_dir, 'epoch-' + str(epoch)))
            config.save_pretrained(os.path.join(args.model_dir, 'epoch-' + str(epoch)))
            with open(os.path.join(args.model_dir, 'epoch-' + str(epoch), 'args.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(vars(args)), f)
            with open(os.path.join(args.log_dir, 'dev-result.txt'), 'w', encoding='utf-8') as f:
                f.write(str(best_val_result))
        gc.collect()
        torch.cuda.empty_cache()
    if args.local_rank in [-1, 0]:
        shutil.copy(os.path.join(args.model_dir, 'epoch-' + str(best_val_epoch), 'args.json'), os.path.join(args.best_model_dir, 'args.json'))
        shutil.copy(os.path.join(args.model_dir, 'epoch-' + str(best_val_epoch), 'config.json'), os.path.join(args.best_model_dir, 'config.json'))
        shutil.copy(os.path.join(args.model_dir, 'epoch-' + str(best_val_epoch), 'pytorch_model.bin'), os.path.join(args.best_model_dir, 'pytorch_model.bin'))
        writer.close()
    model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if args.local_rank != -1:
        dist.barrier()

    config = UniTRecConfig.from_pretrained(args.best_model_dir)
    val_model = UniTRecModel.from_pretrained(args.best_model_dir, config=config, dis_scoring=args.dis_loss, ppl_scoring=args.ppl_loss).cuda()
    result_file = os.path.join(args.prediction_dir, 'prediction.txt')
    if args.local_rank == -1:
        auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = inference(args, val_model, test_dataset, result_file, return_scores=True)
        print(Fore.BLUE + ('%s\nAUC = %.2f\nMRR = %.2f\nnDCG@5 = %.2f\nnDCG@10 = %.2f\nHR@1 = %.2f\nHR@5 = %.2f\nHR@10 = %.2f' % (args.task, auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10)) + Fore.RESET)
    else: # distributed inference and aggregation results
        inference(args, val_model, test_dataset, result_file + '-' + str(args.local_rank), return_scores=False)
        dist.barrier()
        if args.local_rank == 0:
            with open(result_file, 'w', encoding='utf-8') as f:
                index = 0
                for i in range(dist.get_world_size()):
                    with open(result_file + '-' + str(i), 'r', encoding='utf-8') as f_:
                        for line in f_:
                            if len(line.strip()) > 0:
                                index += 1
                                result_line = ('' if index == 1 else '\n') + str(index) + line[line.find(' '):].strip('\n')
                                f.write(result_line)
            with open(test_dataset.truth_file, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
                auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = scoring(truth_f, result_f, onehot=True)
                auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = auc * 100, mrr * 100, ndcg5 * 100, ndcg10 * 100, hr1 * 100, hr5 * 100, hr10 * 100 # return percentage scores
            print(Fore.BLUE + ('%s\nAUC = %.2f\nMRR = %.2f\nnDCG@5 = %.2f\nnDCG@10 = %.2f\nHR@1 = %.2f\nHR@5 = %.2f\nHR@10 = %.2f' % (args.task, auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10)) + Fore.RESET)
    if args.local_rank in [-1, 0]:
        with open(os.path.join(args.log_dir, 'test-result.txt'), 'w', encoding='utf-8') as f:
            f.write('Reddit-quote Test\nAUC = %.2f\nMRR = %.2f\nnDCG@5 = %.2f\nnDCG@10 = %.2f\nHR@1 = %.2f\nHR@5 = %.2f\nHR@10 = %.2f' % (auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10))


def inference(args, model, val_dataset, result_file, return_scores):
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) # batch_size must be 1 and shuffle must be False
    assert len(val_dataloader) % ((val_dataset.candidate_num - 1) // val_dataset.VAL_QUOTE_SEGMENT_NUM + 1) == 0
    val_num = len(val_dataloader) // ((val_dataset.candidate_num - 1) // val_dataset.VAL_QUOTE_SEGMENT_NUM + 1)
    val_indices = val_dataset.indices
    assert model.IGNORE_TOKEN_ID == val_dataset.IGNORE_TOKEN_ID
    assert return_scores == (args.local_rank == -1)
    if args.ppl_eval != args.dis_eval:
        scores = torch.zeros([len(val_indices)]).cuda()
    else:
        ppl_scores = torch.zeros([len(val_indices)]).cuda()
        dis_scores = torch.zeros([len(val_indices)]).cuda()
    index = 0
    model.eval()
    with torch.no_grad():
        for history_input_ids, history_segment_ids, history_global_attention_mask, history_local_position_ids, candidate_input_ids, candidate_cls_indices, targets in val_dataloader:
            history_input_ids = history_input_ids.cuda(non_blocking=True)
            history_segment_ids = history_segment_ids.cuda(non_blocking=True)
            history_global_attention_mask = history_global_attention_mask.cuda(non_blocking=True)
            history_local_position_ids = history_local_position_ids.cuda(non_blocking=True)
            candidate_input_ids = candidate_input_ids.squeeze(dim=0).cuda(non_blocking=True)
            candidate_cls_indices = candidate_cls_indices.squeeze(dim=0).cuda(non_blocking=True)
            targets = targets.squeeze(dim=0).cuda(non_blocking=True)
            candidate_num = candidate_input_ids.size(0)
            _ppl_scores, _dis_scores = model(history_input_ids, history_segment_ids, history_global_attention_mask, history_local_position_ids, candidate_input_ids, candidate_cls_indices, targets)
            if args.ppl_eval and not args.dis_eval:
                scores[index: index+candidate_num] = _ppl_scores
            elif not args.ppl_eval and args.dis_eval:
                scores[index: index+candidate_num] = _dis_scores
            else:
                ppl_scores[index: index+candidate_num] = _ppl_scores
                dis_scores[index: index+candidate_num] = _dis_scores
            index += candidate_num
    if args.ppl_eval != args.dis_eval:
        scores = scores.tolist()
        assert index == len(scores)
        sub_scores = [[] for _ in range(val_num)]
        for i, index in enumerate(val_indices):
            sub_scores[index].append([scores[i], len(sub_scores[index])])
        for i in range(val_num):
            sub_scores[i].sort(key=lambda x: x[0], reverse=True)
        rankings = [[0 for _ in range(len(sub_score))] for sub_score in sub_scores]
        for i, sub_score in enumerate(sub_scores):
            for j in range(len(sub_score)):
                rankings[i][sub_score[j][1]] = j + 1
    else:
        ppl_scores = ppl_scores.tolist()
        dis_scores = dis_scores.tolist()
        assert index == len(ppl_scores) and index == len(dis_scores)
        sub_ppl_scores = [[] for _ in range(val_num)]
        sub_dis_scores = [[] for _ in range(val_num)]
        for i, index in enumerate(val_indices):
            sub_ppl_scores[index].append(ppl_scores[i])
            sub_dis_scores[index].append(dis_scores[i])
        rankings = [AverageRanking.rank(sub_ppl_scores[i], sub_dis_scores[i]) for i in range(val_num)]
    write_predictions(result_file, rankings)
    if return_scores:
        with open(val_dataset.truth_file, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = scoring(truth_f, result_f, onehot=True)
            return auc * 100, mrr * 100, ndcg5 * 100, ndcg10 * 100, hr1 * 100, hr5 * 100, hr10 * 100 # return percentage scores


def test(args):
    config = UniTRecConfig.from_pretrained(args.test_model_path)
    model = UniTRecModel.from_pretrained(args.test_model_path, config=config, dis_scoring=args.dis_loss, ppl_scoring=args.ppl_loss).cuda()
    result_file = os.path.join(args.test_model_path.strip('/').replace('ckpt_models/', 'predictions/') + '-' + args.timestamp + '-test.txt')
    if args.local_rank == -1:
        val_dataset = QuoteRecValDataset(args, mode='test')
        auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = inference(args, model, val_dataset, result_file, return_scores=True)
        print('Test : %s\nAUC = %.2f\nMRR = %.2f\nnDCG@5 = %.2f\nnDCG@10 = %.2f\nHR@1 = %.2f\nHR@5 = %.2f\nHR@10 = %.2f' % (args.test_model_path, auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10))
    else:
        val_dataset = QuoteRecValDataset(args, mode='test', rank=args.local_rank, world_size=dist.get_world_size())
        inference(args, model, val_dataset, result_file + '-' + str(args.local_rank), return_scores=False)
        dist.barrier()
        if args.local_rank == 0:
            with open(result_file, 'w', encoding='utf-8') as f:
                index = 0
                for i in range(dist.get_world_size()):
                    with open(result_file + '-' + str(i), 'r', encoding='utf-8') as f_:
                        for line in f_:
                            if len(line.strip()) > 0:
                                index += 1
                                result_line = ('' if index == 1 else '\n') + str(index) + line[line.find(' '):].strip('\n')
                                f.write(result_line)
            with open(val_dataset.truth_file, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
                auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = scoring(truth_f, result_f, onehot=True)
                auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10 = auc * 100, mrr * 100, ndcg5 * 100, ndcg10 * 100, hr1 * 100, hr5 * 100, hr10 * 100 # return percentage scores
            print('Test : %s\nAUC = %.2f\nMRR = %.2f\nnDCG@5 = %.2f\nnDCG@10 = %.2f\nHR@1 = %.2f\nHR@5 = %.2f\nHR@10 = %.2f' % (args.test_model_path, auc, mrr, ndcg5, ndcg10, hr1, hr5, hr10))


if __name__ == '__main__':
    args = parse_config()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise Exception('Unexpected mode : ' + args.mode)
