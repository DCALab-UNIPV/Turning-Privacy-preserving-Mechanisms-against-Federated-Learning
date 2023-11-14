import torch
import os
import numpy as np
import torch.nn as nn
import dgl
from random import sample
from multiprocessing import Pool, Manager
from collections import defaultdict
# from torch.multiprocessing import Pool, Manager
from model import model
import pdb
import functools
import malicious
from joblib import Parallel, delayed
torch.multiprocessing.set_sharing_strategy('file_system')
import hdbscan
from torchmetrics.functional import pairwise_cosine_similarity
import copy
from sklearn.metrics.pairwise import pairwise_distances
from scipy.special import logit
import sklearn.metrics.pairwise as smp

class server():
    def __init__(self, user_list, user_batch, users, items, embed_size, lr, device, rating_max, rating_min, weight_decay, mal_prop, attack, agg, back_target):
        self.user_list_with_coldstart = user_list
        self.user_list = self.generate_user_list(self.user_list_with_coldstart)
        self.batch_size = user_batch if not user_batch == -1 else len(self.user_list)
        self.user_embedding = torch.randn(len(users), embed_size).to(device).share_memory_()
        self.item_embedding = torch.randn(len(items), embed_size).to(device).share_memory_()
        self.model = model(embed_size).to(device)
        self.lr = lr
        self.rating_max = rating_max
        self.rating_min = rating_min
        self.distribute(self.user_list_with_coldstart)
        self.weight_decay = weight_decay
        self.mal_prop=mal_prop
        self.device = device
        self.attacker = malicious.DriftAttack(1.5)
        self.attack = attack
        self.agg = agg
        self.back_target = back_target
        self.clip = 0
        self.epsilon = 1 
        self.delta = 10e-5
        self.grad_history = np.array([])
        self.alpha = 0

    def _krum_create_distances(self, users_grads):
        distances = defaultdict(dict)
        for i in range(len(users_grads)):
            for j in range(i):
                distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j])
        return distances

    def krum(self, users_grads, users_count, corrupted_count, distances=None,return_index=False, debug=False, multi=False):
        if not return_index:
            assert users_count >= 2*corrupted_count + 1,('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
        non_malicious_count = users_count - corrupted_count
        number_to_consider = non_malicious_count - 2
        minimal_error = 1e20
        minimal_error_index = -1
        errors_dict = {}
        if distances is None:
            distances = self._krum_create_distances(users_grads)
        for user in distances.keys():
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_malicious_count])
            errors_dict[user] = current_error
            if current_error < minimal_error:
                minimal_error = current_error
                minimal_error_index = user

        errors_dict_sorted = dict(sorted(errors_dict.items(), key=lambda item: item[1], reverse=True))
        indexes = [*errors_dict_sorted][:number_to_consider]
        if return_index:
            return minimal_error_index
        else:
            if multi:
                return np.mean(users_grads[indexes], axis=0)
            else:
                return users_grads[minimal_error_index]
    def trimmed_mean(self, gradient_model, corrupted_count, current_grads=None, attack_model=None):
        number_to_consider = int(gradient_model.shape[0] - corrupted_count) - 1
        current_model_grads = np.empty((gradient_model.shape[1],), gradient_model.dtype)
        
        for i, param_across_users in enumerate(gradient_model.T):
            med = np.median(param_across_users)
            good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
            current_model_grads[i] = np.mean(good_vals) + med
        return current_model_grads

    def flame(self, gradient_model, num_clients):
        tensor_gradient_model = torch.from_numpy(gradient_model)
        cos_list = pairwise_cosine_similarity(tensor_gradient_model, zero_diagonal=False)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
        benign_client = []
        norm_list = np.array([])

        max_num_in_cluster=0
        max_cluster_index=0
        if clusterer.labels_.max() < 0:
            for i in range(len(local_model)):
                benign_client.append(i)
                norm_list = np.append(norm_list, torch.norm(tensor_gradient_model[i],p=2).item())
        else:
            for index_cluster in range(clusterer.labels_.max()+1):
                if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
            for i in range(len(clusterer.labels_)):
                if clusterer.labels_[i] == max_cluster_index:
                    benign_client.append(i)
                    norm_list = np.append(norm_list, torch.norm(tensor_gradient_model[i],p=2).item())

        self.clip_value = np.median(norm_list)
        for i in range(len(benign_client)):
            gama = self.clip_value/norm_list[i]
            if gama < 1:
                gradient_model[benign_client[i]] *= gama
        return np.mean(gradient_model[benign_client], axis=0)

    def foolsgold(self, grad_history, grad_in, num_workers):
        epsilon = 1e-5
        
        if grad_history.shape[0] != num_workers:
            grad_history = grad_history[:num_workers,:] + grad_history[num_workers:,:]

        similarity_maxtrix = smp.cosine_similarity(grad_history) - np.eye(num_workers)

        mv = np.max(similarity_maxtrix, axis=1) + epsilon

        alpha = np.zeros(mv.shape)
        for i in range(num_workers):
            for j in range(num_workers):
                if mv[j] > mv[i]:
                    similarity_maxtrix[i,j] *= mv[i]/mv[j]

        alpha = 1 - (np.max(similarity_maxtrix, axis=1))
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0
        alpha = alpha/np.max(alpha)
        alpha[(alpha == 1)] = 0.99
        alpha = (np.log((alpha / (1 - alpha)) + epsilon) + 0.5)
        alpha[(np.isinf(alpha) + alpha > 1)] = 1
        alpha[(alpha < 0)] = 0
        
        grad = np.average(grad_in, weights=alpha, axis=0).astype(np.float32)
        
        return grad, grad_history, alpha


    def generate_user_list(self, user_list_with_coldstart):
        ls = []
        for user in user_list_with_coldstart:
            if len(user.items) > 0:
                ls.append(user)
        return ls
    
    def row_into_parameters(self, row, parameters):
        offset = 0
        new_grads = []
        for param in parameters:
            new_size = functools.reduce(lambda x,y:x*y, param.shape)
            current_data = row[offset:offset + new_size]

            new_grads.append(torch.from_numpy(current_data.reshape(param.shape)).to(self.device))
            offset += new_size
        return new_grads

    def aggregator(self, parameter_list, corrupted_count, step):
        flag = False
        number = 0
        gradient_item = torch.zeros_like(self.item_embedding).to(self.device)
        gradient_user = torch.zeros_like(self.user_embedding).to(self.device)
        loss = 0
        loss_mal = 0
        mal_count = 0
        item_count = torch.zeros(self.item_embedding.shape[0], device=self.device)
        user_count = torch.zeros(self.user_embedding.shape[0], device=self.device)
        mal_involved = 1e-10
        for parameter in parameter_list:
            [model_grad, item_grad, user_grad, returned_items, returned_users, loss_user, loss_user_mal, mal] = parameter
            num = len(returned_items)
            item_count[returned_items] += 1
            user_count[returned_users] += num
            loss += loss_user ** 2 * num
            loss_mal += loss_user_mal ** 2 
            if not loss_user_mal == 0:
                mal_involved+=1
            number += num
            if not flag:
                flag = True
                gradient_model = []
                gradient_model_mal = []
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                if not mal:
                    gradient_model.append(np.concatenate([grads.data.cpu().numpy().flatten() for grads in model_grad]))
                else:
                    mal_count+=1
                    gradient_model_mal.append(np.concatenate([grads.data.cpu().numpy().flatten() for grads in model_grad]))
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                if not mal:
                    gradient_model.append(np.concatenate([grads.data.cpu().numpy().flatten() for grads in model_grad]))
                else:
                    mal_count+=1
                    gradient_model_mal.append(np.concatenate([grads.data.cpu().numpy().flatten() for grads in model_grad]))

        if self.attack == "LIE" or self.attack == "both":
            attackers = int(self.batch_size*0.24)
            gradient_model_mal[-attackers:] = self.attacker.attack(gradient_model_mal[-attackers:])

        gradient_model = gradient_model + gradient_model_mal

        if type(loss_mal) == int:
            loss_mal = torch.tensor(0)
        loss = torch.sqrt(loss / number)
        loss_mal = torch.sqrt(loss_mal / mal_involved)
        print('step', step, 'trianing average loss:', loss, 'trianing average mal loss:', loss_mal, 'mal count:', mal_count)
        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1
        gradient_item /= item_count.unsqueeze(1)
        gradient_user /= user_count.unsqueeze(1)

        #print(gradient_model[0].shape)
        if self.agg == "TrimmedMean":
            current_model_grads = self.trimmed_mean(np.array(gradient_model), corrupted_count)
        elif self.agg == "Krum":
            current_model_grads = self.krum(np.array(gradient_model), self.batch_size ,corrupted_count)
        elif self.agg == "Flame":
            current_model_grads = self.flame(np.array(gradient_model), self.batch_size)
        elif self.agg == "Foolsgold":
            gradient_model = np.array(gradient_model)
            if self.grad_history.size == 0:
                self.grad_history = gradient_model
            else:
                self.grad_history = np.concatenate((self.grad_history, gradient_model), axis=0)
            #print(self.grad_history.shape)
            current_model_grads, self.grad_history, self.alpha = self.foolsgold(self.grad_history, gradient_model, self.batch_size)
        
        return self.row_into_parameters(current_model_grads, self.model.parameters()), gradient_item, gradient_user

    def distribute(self, users):
        for user in users:
            user.update_local_GNN(self.model, self.rating_max, self.rating_min, self.user_embedding, self.item_embedding)

    def distribute_one(self, user):
        user.update_local_GNN(self.model, self.rating_max, self.rating_min, self.user_embedding, self.item_embedding)

    def predict(self, valid_data):
        
        users = valid_data[:, 0]
        items = valid_data[:, 1]
        res = []
        self.distribute([self.user_list_with_coldstart[i] for i in set(users)])

        for i in range(len(users)):
            res_temp = self.user_list_with_coldstart[users[i]].predict(items[i], self.user_embedding, self.item_embedding)
            res.append(float(res_temp))
        return np.array(res)

    def predict_backdoor_all(self, valid_data, target_user):
        
        users = valid_data[:, 0]
        items = valid_data[:, 1]
        res = []
        self.distribute([self.user_list_with_coldstart[i] for i in set(users)])

        for i in range(len(users)):
            res_temp = self.user_list_with_coldstart[users[target_user]].predict(items[i], self.user_embedding, self.item_embedding)
            res.append(float(res_temp))
        return np.array(res)

    def predict_backdoor(self, data, target_user, backdoor_samples):
        
        res = []
        self.distribute_one(self.user_list_with_coldstart[target_user])

        res_temp = self.user_list_with_coldstart[target_user].predict(torch.tensor(backdoor_samples), self.user_embedding, self.item_embedding)
        return np.array(res_temp)

    def train_one(self, user, user_embedding, item_embedding):
        print(user)
        self.parameter_list.append(user.train(user_embedding, item_embedding))

    def train(self, step):
        parameter_list = []
        users = sample(self.user_list, self.batch_size)
        
        self.distribute(users)
        
        for user in users:
            parameter_list.append(user.train(self.user_embedding, self.item_embedding, self.back_target))
        

        gradient_model, gradient_item, gradient_user = self.aggregator(parameter_list, int(len(users)*self.mal_prop),step)

        ls_model_param = list(self.model.parameters())

        item_index = gradient_item.sum(dim = -1) != 0
        user_index = gradient_user.sum(dim = -1) != 0
        
        for i in range(len(ls_model_param)):
            ls_model_param[i].data = ls_model_param[i].data - self.lr * gradient_model[i] - self.weight_decay * ls_model_param[i].data
            if self.agg == "Flame":
                ls_model_param[i].data += np.random.normal(loc=0, scale=0.001*self.clip_value)
        self.item_embedding[item_index] = self.item_embedding[item_index] -  self.lr * gradient_item[item_index] - self.weight_decay * self.item_embedding[item_index]
        self.user_embedding[user_index] = self.user_embedding[user_index] -  self.lr * gradient_user[user_index] - self.weight_decay * self.user_embedding[user_index]
