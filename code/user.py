import torch
import copy
from random import sample
import torch.nn as nn
import numpy as np
import dgl
import pdb
from model import model

class user():
    def __init__(self, id_self, items, ratings, neighbors, embed_size, clip, laplace_lambda, negative_sample, negative_sample_mal, mal, attack_epochs, device, attack, pseudo, backdoor_samples, target_user, rating_min):
        self.negative_sample = negative_sample
        self.negative_sample_mal = negative_sample_mal
        self.clip = clip
        self.laplace_lambda = laplace_lambda
        self.id_self = id_self
        self.items = items
        self.embed_size = embed_size
        self.ratings = ratings
        self.neighbors = neighbors
        self.model = model(embed_size).to(device)
        self.graph = self.build_local_graph(id_self, items, neighbors, device)
        self.graph = dgl.add_self_loop(self.graph)
        self.user_feature = torch.randn(self.embed_size)
        self.mal = mal
        self.attack_item_embed = torch.randn((negative_sample_mal, embed_size), requires_grad=True, device=device)
        self.optimizer = torch.optim.AdamW([self.attack_item_embed], lr=0.001)
        self.mal_loss = 0
        self.attack_epochs = attack_epochs
        self.device = device
        self.attack = attack
        self.pseudo = pseudo
        self.backdoor_samples = backdoor_samples
        self.target_user = target_user
        self.rating_min = rating_min

    def build_local_graph(self, id_self, items, neighbors, device):
        G = dgl.DGLGraph()
        dic_user = {self.id_self: 0}
        dic_item = {}
        count = 1
        for n in neighbors:
            dic_user[n] =  count
            count += 1
        for item in items:
            dic_item[item] = count
            count += 1
        G.add_edges([i for i in range(1, len(dic_user))], 0)
        G.add_edges(list(dic_item.values()), 0)
        G.add_edges(0, 0)
        G = G.to(device)
        return G

    def user_embedding(self, embedding):
        return embedding[torch.tensor(self.neighbors)], embedding[torch.tensor(self.id_self)]

    def target_user_embedding(self, embedding):
        return embedding[torch.tensor(self.target_user)]

    def item_embedding(self, embedding):
        return embedding[torch.tensor(self.items)]

    def GNN_train_attack(self, embedding_user, embedding_item, local_model):
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        neighbor_embedding = neighbor_embedding.detach()
        self_embedding = self_embedding.detach()
        items_embedding = self.item_embedding(embedding_item).detach()
        items_embedding_with_sampled = torch.cat((items_embedding, self.attack_item_embed), dim = 0)
        user_feature = local_model(self_embedding, neighbor_embedding, items_embedding)
        predicted = torch.matmul(user_feature, items_embedding_with_sampled.t())
        return predicted

    def GNN(self, embedding_user, embedding_item, sampled_items, target_user=None):
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        items_embedding = self.item_embedding(embedding_item)
        items_embedding.to(self.device)
        if (not self.mal or self.attack == "LIE") and self.pseudo == 'True':
            sampled_items_embedding = embedding_item[torch.tensor(sampled_items)]
        elif self.pseudo == 'False':
            sampled_items_embedding = torch.tensor([], device=self.device)
        elif self.mal and self.attack == "backdoor":
            sampled_items_embedding = embedding_item[torch.tensor(self.backdoor_samples)]
            target_user_embedding = self.target_user_embedding(embedding_user)
        else:
            sampled_items_embedding = self.attack_item_embed.detach()
        sampled_items_embedding.to(self.device)
        
        if self.mal and self.attack == "backdoor":
            user_feature = self.model(self_embedding, neighbor_embedding, items_embedding)
            target_user_feature = self.model(target_user_embedding, neighbor_embedding, sampled_items_embedding)
            predicted_user = torch.matmul(user_feature, items_embedding.t())
            self.user_feature = user_feature.detach()
            predicted_target_user = torch.matmul(target_user_feature, sampled_items_embedding.t())
            predicted = torch.cat((predicted_user, predicted_target_user))
        else:
            items_embedding_with_sampled = torch.cat((items_embedding, sampled_items_embedding), dim = 0)
            user_feature = self.model(self_embedding, neighbor_embedding, items_embedding)
            predicted = torch.matmul(user_feature, items_embedding_with_sampled.t())
            self.user_feature = user_feature.detach()
        return predicted

    def update_local_GNN(self, global_model, rating_max, rating_min, embedding_user, embedding_item):
        self.model = copy.deepcopy(global_model)
        self.rating_max = rating_max
        self.rating_min = rating_min
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        if len(self.items) > 0:
            items_embedding = self.item_embedding(embedding_item)
        else:
            items_embedding = False
        user_feature = self.model(self_embedding, neighbor_embedding, items_embedding)
        self.user_feature = user_feature.detach()

    def loss(self, predicted, sampled_rating):
        if self.pseudo == 'True':
            true_label = torch.cat((torch.tensor(self.ratings).to(sampled_rating.device), sampled_rating))
        else:
            true_label = torch.tensor(self.ratings).to(self.device)
        return torch.sqrt(torch.mean((predicted - true_label) ** 2))

    def predict(self, item_id, embedding_user, embedding_item):
        self.model.eval()
        item_embedding = embedding_item[item_id]
        return torch.matmul(self.user_feature, item_embedding.t())

    def train_attack(self, embedding_user, embedding_item):
        embedding_user = torch.clone(embedding_user).detach().to(self.device)
        embedding_item = torch.clone(embedding_item).detach().to(self.device)
        embedding_user.requires_grad = False
        embedding_item.requires_grad = False
        embedding_user.grad = torch.zeros_like(embedding_user, device=self.device)
        embedding_item.grad = torch.zeros_like(embedding_item, device=self.device)

        local_model = copy.deepcopy(self.model).to(self.device)
        local_model.eval()

        alpha = -1
        for _ in range(self.attack_epochs):
            sampled_rating = torch.matmul(self.user_feature, self.attack_item_embed.t())
            sampled_rating = torch.round(torch.clip(sampled_rating, min = self.rating_min, max = self.rating_max))
            predicted = self.GNN_train_attack(embedding_user, embedding_item, local_model)
            loss = alpha * self.loss(predicted, sampled_rating)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.mal_loss = loss.detach()

    def negative_sample_item(self, embedding_user, embedding_item):
        item_num = embedding_item.shape[0]
        ls = [i for i in range(item_num) if i not in self.items]
        if (not self.mal or self.attack == "LIE" or self.attack == "backdoor") and self.pseudo == 'True':
            sampled_items = sample(ls, self.negative_sample)
            sampled_item_embedding = embedding_item[torch.tensor(sampled_items)]
        elif self.pseudo == 'False':
            return [], []
        else:
            sampled_items = []
            self.train_attack(embedding_user, embedding_item)
            sampled_item_embedding = torch.clone(self.attack_item_embed).detach()
        predicted = torch.matmul(self.user_feature, sampled_item_embedding.t())
        predicted = torch.round(torch.clip(predicted, min = self.rating_min, max = self.rating_max))

        return sampled_items, predicted

    def LDP(self, tensor):
        tensor_mean = torch.abs(torch.mean(tensor)).cpu()
        tensor = torch.clamp(tensor, min = -self.clip, max = self.clip)
        noise = np.random.laplace(0, tensor_mean * self.laplace_lambda)
        tensor += noise
        return tensor

    def train(self, embedding_user, embedding_item, rating):
        embedding_user = torch.clone(embedding_user).detach().to(self.device)
        embedding_item = torch.clone(embedding_item).detach().to(self.device)
        embedding_user.requires_grad = True
        embedding_item.requires_grad = True
        embedding_user.grad = torch.zeros_like(embedding_user)
        embedding_item.grad = torch.zeros_like(embedding_item)

        self.model.train()
        sampled_items, sampled_rating = self.negative_sample_item(embedding_user, embedding_item)
        if self.mal and self.attack == "backdoor":
            sampled_rating = torch.ones_like(sampled_rating) * rating
            returned_items = self.items + self.backdoor_samples
            predicted = self.GNN(embedding_user, embedding_item, sampled_items)
        else:
            returned_items = self.items + sampled_items
            predicted = self.GNN(embedding_user, embedding_item, sampled_items)
        loss = self.loss(predicted, sampled_rating)
        self.model.zero_grad()
        loss.backward()
        model_grad = []
        for param in list(self.model.parameters()):
            if self.pseudo == "False":
                grad = param.grad
            else:
                grad = self.LDP(param.grad)
            model_grad.append(grad)

        item_grad = self.LDP(embedding_item.grad[returned_items, :])
        
        if self.mal and self.attack == "backdoor":
            returned_users = self.neighbors + [self.id_self, self.target_user]
        else:
            returned_users = self.neighbors + [self.id_self]
        user_grad = self.LDP(embedding_user.grad[returned_users, :])
        res = (model_grad, item_grad, user_grad, returned_items, returned_users, loss.detach(), self.mal_loss, self.mal)
        return res