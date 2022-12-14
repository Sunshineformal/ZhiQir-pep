import os
import math
import torch, gc
import argparse
from transformers import GPT2Tokenizer, AdamW
from module import ContinuousPromptLearning
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
import torch.nn as nn
import pickle
# -*- encoding:utf-8 -*-

import json
import time
parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')
parser.add_argument('--data_path', type=str, default='./CIKM20-NETE-Datasets/Amazon/MoviesAndTV/reviews.pickle',
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default='./CIKM20-NETE-Datasets/Amazon/MoviesAndTV/1',
                    help='load indexes')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',default='cuda:1',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./pepler/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')



print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
corpus = DataLoader(args.data_path, args.index_dir, tokenizer, args.words)
feature_set = corpus.feature_set
train_data = Batchify(corpus.train, tokenizer, bos, eos, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, tokenizer, bos, eos, args.batch_size)
test_data = Batchify(corpus.test, tokenizer, bos, eos, args.batch_size)#???????????????,???train?????????????????????

nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntoken = len(tokenizer)
model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)
model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table
model.to(device)

def generate(data,user_id,item_id):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
            user, item, rating, seq, mask = data.search(user_id,item_id) # data.step += 1
            user_embeddings = nn.Embedding(nuser, 768)
            item_embeddings = nn.Embedding(nitem, 768)
            u_src = user_embeddings(user)  # (batch_size, emsize)
            i_src = item_embeddings(item)  # (batch_size, emsize)
            rating = torch.sum(u_src * i_src, 1)
            # user, item, rating, seq, mask = data.next_batch()
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            # seq.view(1,25)
            mask.to(device)
            text = seq[:, :1].to(device)  # bos, (batch_size, 1)

            for idx in range(seq.size(1)):
                # produce a word at each step
                outputs = model(user, item, text, None)
                # print(rating)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)

    return idss_predict,rating

with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


def gson(user_id):
        inputef = open('./CIKM20-NETE-Datasets/Amazon/MoviesAndTV/reviews.pickle','rb')
        info = pickle.load(inputef)
        filejson = open('./CIKM20-NETE-Datasets/Amazon/MoviesAndTV/item.json','r')
        injson = json.load(filejson)
        info.sort(key=lambda x:x['user'],reverse=False)                #????????????
        user_name = corpus.user_dict.idx2entity[int(user_id)]
        searched_list = []
        tosort_list = []
        for i in range(len(info)):
            if(info[i]['user']==user_name):
                searched_list.append(info[i])                                #????????????
        for j in range(len(searched_list)):
            item_id = str(corpus.item_dict.entity2idx[searched_list[j]['item']])
            idss_predicted,rating_o = generate(train_data, user_id, item_id)
            tokens_test = [ids2tokens(ids[1:], tokenizer, eos) for ids in test_data.seq.tolist()]
            tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
            text_test = [' '.join(tokens) for tokens in tokens_test]
            text_predict = [' '.join(tokens) for tokens in tokens_predict]
            text_o, rating_o = generate(train_data, user_id, item_id)
            text_out = ''
            for (real, fake) in zip(text_test, text_predict):
                text_out += '{}\n{}\n\n'.format(real, fake)
            if 'imUrl' in injson[corpus.item_dict.entity2idx[searched_list[j]['item']]]:
                image = injson[corpus.item_dict.entity2idx[searched_list[j]['item']]]['imUrl']
            else:
                image = None
            if 'title' in injson[corpus.item_dict.entity2idx[searched_list[j]['item']]]:
                    tosort_list.append({'user': searched_list[j]['user'],
                                        'item': injson[corpus.item_dict.entity2idx[searched_list[j]['item']]]['title'],
                                        'text_out': text_out, 'rating': round(rating_o.item(),3),'image':image})
            else:
                    tosort_list.append({'user': searched_list[j]['user'], 'item': searched_list[j]['item'], 'text_out': text_out,
                                        'rating':  round(rating_o.item(),3),'image':image})

        tosort_list.sort(key=lambda x:x['rating'],reverse=True)
        for t in range(len(tosort_list)):
            print(tosort_list[t])
        print(len(tosort_list))


        with open("d:\\peplerdata"+str(user_id) + ".json", "w") as f:
            f.write("{ ")
            for j in range(len(tosort_list)):
                print("??????" + tosort_list[j]['user'] + "???" + str(tosort_list[j]['item']))
                f.write(" \"" + "user" + str(j) + "\":\"" + str(tosort_list[j]['user']) + "\"" +
                        " , \"" + "item" + str(j) + "\":" + "\"" + str(tosort_list[j]['item']) + "\"" +
                        " , \"" + "out" + str(j) + "\":" + "\"" + str(tosort_list[j]['text_out']).replace("\n",",") + "\""+
                        " , \"" + "rating" + str(j) + "\":" + "\"" + str(tosort_list[j]['rating']) + "\"" +
                        " , \"" + "image" + str(j) + "\":" + "\"" + str(tosort_list[j]['image']) + "\""
                        )
                if j != len(tosort_list)-1:
                    f.write(",")
            f.write("}")

import socket
import sys
import threading
import json
import numpy as np


def main():
    # ????????????????????????
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # ????????????????????????
    host = socket.gethostname()
    # ??????????????????
    port = 12345
    # ??????????????????????????????????????????
    serversocket.bind((host, port))
    # ???????????????????????????
    serversocket.listen(5)
    # ????????????????????????????????????
    myaddr = serversocket.getsockname()
    print("???????????????:%s" % str(myaddr))
    # ?????????????????????????????????
    while True:
        # ???????????????????????????
        clientsocket, addr = serversocket.accept()
        print("????????????:%s" % str(addr))
        try:
            t = ServerThreading(clientsocket)  # ??????????????????????????????????????????
            t.start()
            pass
        except Exception as identifier:
            print(identifier)
            pass
        pass
    serversocket.close()
    pass


class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self, clientsocket, recvsize=1024 * 1024, encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("????????????.....")
        try:
            # ????????????
            msg = ''
            while True:
                # ??????recvsize?????????
                rec = self._socket.recv(self._recvsize)
                #print(rec)
                # ??????
                msg += rec.decode(self._encoding)
                #print(msg)
                # ?????????????????????????????????python socket?????????????????????????????????????????????
                # ???????????????????????????????????????????????????
                if msg.strip().endswith('over'):
                    msg = msg[:-4]
                    break
            # ??????json???????????????
            re = json.loads(msg)
            #print(re)
            print("??????????????????"+str(re)+"????????????gson("+str(re)+")")
            gson(str(re))
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("????????????.....")

        pass

    def __del__(self):

        pass

if __name__ == "__main__":
    main()