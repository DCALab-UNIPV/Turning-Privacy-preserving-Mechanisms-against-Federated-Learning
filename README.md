# Turning Privacy preserving Mechanisms against Federated Learning
Offical code for the paper "Turning Privacy-preserving Mechanisms against Federated Learning" accepted at ACM Conference on Computer and Communications Security (CCS) 2023

## Authors: Marco Arazzi, Mauro Conti, Antonino Nocera and Stjepan Picek

## Abstract:
Recently, researchers have successfully employed Graph Neural Networks (GNNs) to build enhanced recommender systems due to their capability to learn patterns from the interaction between involved entities.
In addition, previous studies have investigated federated learning as the main solution to enable a native privacy-preserving mechanism for the construction of global GNN models without collecting sensitive data into a single computation unit.
Still, privacy issues may arise as the analysis of local model updates produced by the federated clients can return information related to sensitive local data.
For this reason, researchers proposed solutions that combine federated learning with Differential Privacy strategies and community-driven approaches, which involve combining data from neighbor clients to make the individual local updates less dependent on local sensitive data.
  
In this paper, we identify a crucial security flaw in such a configuration and  design an attack capable of deceiving state-of-the-art defenses for federated learning.
The proposed attack includes two operating modes, the first one focusing on convergence inhibition (Adversarial Mode), and the second one aiming at building a deceptive rating injection on the global federated model (Backdoor Mode).  
The experimental results show the effectiveness of our attack in both its modes, returning on average 60% performance detriment in all the tests on Adversarial Mode and fully effective backdoors in 93% of cases for the tests performed on

## Link: 
- [Turning Privacy preserving Mechanisms against Federated Learning (Arxiv)](https://arxiv.org/abs/2305.05355)


## Cite

    @article{arazzi2023turning,
        title={Turning Privacy-preserving Mechanisms against Federated Learning},
        author={Arazzi, Marco and Conti, Mauro and Nocera, Antonino and Picek, Stjepan},
        journal={arXiv preprint arXiv:2305.05355},
        year={2023}
    }
