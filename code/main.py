import pickle
import torch
import numpy as np
from user import user
from server import server
from sklearn import metrics
import random
import math
import argparse
import warnings
import sys
import faulthandler
from random import sample
import pickle
import os
faulthandler.enable()
warnings.filterwarnings('ignore')
# torch.multiprocessing.set_sharing_strategy('file_system')
#random.seed(123)

parser = argparse.ArgumentParser(description="args")
parser.add_argument('--embed_size', type=int, default=8)
parser.add_argument('--lr', type=float, default = 0.01)
parser.add_argument('--data', default='ciao', choices=['ciao', 'filmtrust', 'epinions'])
parser.add_argument('--user_batch', type=int, default=256)
parser.add_argument('--clip', type=float, default = 0.3)
parser.add_argument('--laplace_lambda', type=float, default = 0.1)
parser.add_argument('--negative_sample', type = int, default = 10)
parser.add_argument('--negative_sample_mal', type = int, default = 10)
parser.add_argument('--mal_prop', type = float, default = 0.3)
parser.add_argument('--atk_eps', type = int, default = 200)
parser.add_argument('--valid_step', type = int, default = 20)
parser.add_argument('--weight_decay', type = float, default = 0.001)
parser.add_argument('--device', type = str, default = 'cpu')
parser.add_argument('--attack', type = str, default = 'ours', choices=['ours', 'backdoor'])
parser.add_argument('--pseudo', type = str, default='True')
parser.add_argument('--aggregator', type = str, default='TrimmedMean', choices=['TrimmedMean','Krum', 'Foolsgold', 'Flame'])
parser.add_argument('--back_rating', type = str, default='min', choices=['min', 'max'])
parser.add_argument('--back_samples_mode', type = str, default='random',  choices=['random', 'min_max'])
args = parser.parse_args()
print(args)

if args.back_samples_mode == 'min_max':
    random.seed(1234)

embed_size = args.embed_size
user_batch = args.user_batch
lr = args.lr
device = torch.device('cpu')
if args.device != 'cpu':
    device = torch.device('cuda:0')

def success_rate_with_threshold(max_rating, preds):
    over_5 = 0
    over_6 = 0
    over_7 = 0
    over_8 = 0
    over_9 = 0
    if args.back_rating == "min":
        for p in preds:
            if p < max_rating*0.5: over_5+=1
            if p < max_rating*0.4: over_6+=1
            if p < max_rating*0.3: over_7+=1
            if p < max_rating*0.2: over_8+=1
            if p < max_rating*0.1: over_9+=1
    else:
        for p in preds:
            if p > max_rating*0.5: over_5+=1
            if p > max_rating*0.6: over_6+=1
            if p > max_rating*0.7: over_7+=1
            if p > max_rating*0.8: over_8+=1
            if p > max_rating*0.9: over_9+=1
    res = [over_5/len(preds), over_6/len(preds), over_7/len(preds), over_8/len(preds), over_9/len(preds)]
    print(f"Success over Threshold: {res}")


def save_ranking(file, ranking):
    ranking_file = open(f'recomandation/{file}', 'wb')
    pickle.dump(ranking, ranking_file)                     
    ranking_file.close()

def load_ranking(file):
    ranking_file = open(f'recomandation/{file}', 'rb')
    ranking = pickle.load(ranking_file)
    ranking_file.close()
    return ranking

def predict_ratings(server, valid_data, data_backdoor ,backdoor_samples, max_rating):
    items = valid_data[:, 1]
    all_items = list(items) + list(backdoor_samples)
    ratings_valid = server.predict_backdoor_all(valid_data, target_user)
    predicted_back = server.predict_backdoor(data_backdoor, target_user, backdoor_samples)
    success_rate_with_threshold(max_rating, predicted_back)
    ratings_pred = list(ratings_valid) + list(predicted_back)
    return all_items, ratings_pred

def sort_ratings(items, ratings):
    ratings_dict = {}
    for item, rating in zip(items, ratings):
        ratings_dict[item] = rating
    ratings_dict_sorted = dict(sorted(ratings_dict.items(), key=lambda item: item[1], reverse=True))
    return [*ratings_dict_sorted]


def rank_difference(backdoor, backdoor_samples):
    rank_diff = 0
    avg_back = 0
    avg_real = 0
    real = load_ranking(f'{args.data}_items_rating.pkl')
    if args.back_rating == 'min':
        fav_count = args.negative_sample
        for b in backdoor_samples:
            avg_back += backdoor.index(b)
            avg_real += real.index(b)
            rank_diff += abs(backdoor.index(b) - real.index(b))
            if backdoor.index(b) < args.negative_sample*2:
                fav_count-=1
    else:
        fav_count = 0
        for b in backdoor_samples:
            avg_back += (backdoor.index(b) +1)
            avg_real += (real.index(b) +1)
            rank_diff += abs((backdoor.index(b) +1) - (real.index(b) +1))
            if backdoor.index(b) < args.negative_sample*5:
                fav_count+=1
    print(f'Avg Back {args.back_rating} ranking {avg_back/args.negative_sample}, Avg real {avg_real/args.negative_sample}')
    return rank_diff/len(backdoor_samples), (rank_diff/len(backdoor_samples))/len(real), fav_count/args.negative_sample


def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                res.append((int(key), int(item), rate))
    return np.array(res)

def loss(server, valid_data, data_backdoor=None, backdoor_flag=False, target_user=None, backdoor_samples=None, rating=0):
    
    def success_rate(res_test_ste, res_back):
        count = 0
        print(res_test_ste)
        print(res_back)
        for r_back in res_back:
            if r_back < res_test_ste:
                count+=1
        return count/len(res_back)

    label_test = valid_data[:, -1]
    predicted_test = server.predict(valid_data)
    mae = sum(abs(label_test - predicted_test)) / len(label_test)
    rmse = math.sqrt(sum((label_test - predicted_test) ** 2) / len(label_test))
    if backdoor_flag:
        label_back = torch.ones(len(backdoor_samples)) * rating
        predicted_back = server.predict_backdoor(data_backdoor, target_user, backdoor_samples)
        mae = sum(abs(label_back - predicted_back)) / len(label_back)
        rmse = math.sqrt(sum((label_back - predicted_back) ** 2) / len(label_back))
        res_test = abs(label_test - predicted_test)
        res_back = abs(label_back - predicted_back)
        res_test_ste = np.std(res_test) #/ math.sqrt(len(res_test))
        s_rate = success_rate(res_test_ste, res_back)
        return mae, rmse, s_rate*100
    return mae, rmse

def mal_list_generator(len_user_list, mal_prop):
    mal_list = []
    mal_num = len_user_list*mal_prop
    for i in range(len_user_list):
        mal_list.append(i<mal_num)
    random.shuffle(mal_list)
    return mal_list

def target_user_samples(user_id, data, item_id_list, negative_sample):
    item_embedding = torch.randn(len(item_id_list), 8).to(device).share_memory_()
    ratings_train = train_data[user_id]
    #ratings_valid = valid_data[u]
    items = []
    rating = []
    for i in range(len(ratings_train)):
        item, rate, _  = ratings_train[i]
        items.append(item)
        rating.append(rate)
    
    if not os.path.exists(f'recomandation/{args.data}_items_rating.pkl') or args.back_samples_mode == 'random':
        item_num = item_embedding.shape[0]
        ls = [i for i in range(item_num) if i not in items]
        sampled_items = sample(ls, negative_sample)
    else:
        real = load_ranking(f'{args.data}_items_rating.pkl')
        #print(real)
        ls = [i for i in real if i not in items]
        if args.back_rating == 'min':
            sampled_items = ls[:args.negative_sample]
        else:
            sampled_items = ls[-args.negative_sample:]
    return sampled_items    

# read data
data_file = open('../data/' + args.data + '.pkl', 'rb')
[train_data, valid_data, test_data, user_id_list, item_id_list, social] = pickle.load(data_file)
data_file.close()
valid_data = processing_valid_data(valid_data)
test_data = processing_valid_data(test_data)

# build user_list
rating_max = -9999
rating_min = 9999
user_list = []
mal_list = mal_list_generator(len(user_id_list), args.mal_prop)
flag_backdoor = True

target_user = 0
backdoor_samples = []

for u, mal in zip(user_id_list, mal_list):
    if not mal:
        target_user = u
        backdoor_samples = target_user_samples(u, train_data, item_id_list, args.negative_sample)
        break

for u, mal in zip(user_id_list, mal_list):
    ratings = train_data[u]
    items = []
    rating = []
    for i in range(len(ratings)):
        item, rate, _  = ratings[i]
        items.append(item)
        rating.append(rate)

    if len(rating) > 0:
        rating_max = max(rating_max, max(rating))
        rating_min = min(rating_min, min(rating))
    user_list.append(user(u, items, rating, list(social[u]), embed_size, args.clip, args.laplace_lambda, args.negative_sample, args.negative_sample_mal, mal, args.atk_eps, args.device, args.attack, args.pseudo, backdoor_samples, target_user, rating_min))

if args.back_rating == 'min':
    back_target = rating_min
else:
    back_target = rating_max


server = server(user_list, user_batch, user_id_list, item_id_list, embed_size, lr, device, rating_max, rating_min, args.weight_decay, args.mal_prop, args.attack, args.aggregator, back_target)
count = 0
step = 0

rmse_best = 9999
valid_rmse_list = []
valid_mae_list = []
while 1:
    for i in range(args.valid_step):
        step+=1
        server.train(step)
    print('valid')
    mae, rmse = loss(server, valid_data)
    valid_rmse_list.append(rmse)
    valid_mae_list.append(mae)
    print('Step: {}, valid mae: {}, valid rmse:{}'.format(step, mae, rmse))
    if args.attack == "backdoor":
        
        mae_back, rmse_back, success_rate = loss(server, test_data, data_backdoor=train_data, backdoor_flag=True, target_user=target_user, backdoor_samples=backdoor_samples, rating=back_target)
        print('Backdoor Step: {}, backdoor mae: {}, backdoor rmse:{}, Success Rate:{}'.format(step, mae_back, rmse_back, success_rate))
    if rmse < rmse_best:
        rmse_best = rmse
        count = 0
        mae_test, rmse_test = loss(server, test_data)
    else:
        count += 1
    if count > 5:
        print('not improved for 5 epochs, stop trianing')
        break

if args.attack == "backdoor":
    itms, rtgs = predict_ratings(server, valid_data, train_data, backdoor_samples, rating_max)
    
    if args.mal_prop == 0.0:
        real_rtgs = sort_ratings(itms, rtgs)
        if not os.path.exists(f'recomandation/{args.data}_items_rating2.pkl'): save_ranking(f'{args.data}_items_rating2.pkl', real_rtgs)
        diff, diff_perc, fav_perc = rank_difference(real_rtgs, backdoor_samples)
        print('Average rank diff: {}, Average perc rank diff: {}, Fav perc {}'.format(diff, diff_perc, fav_perc))
    else:
        back_rtgs = sort_ratings(itms, rtgs)
        diff, diff_perc, fav_perc = rank_difference(back_rtgs, backdoor_samples)
        print('Average rank diff: {}, Average perc rank diff: {}, Fav perc {}'.format(diff, diff_perc, fav_perc))


print(args)
print('final test mae: {}, test rmse: {}'.format(mae_test, rmse_test))

