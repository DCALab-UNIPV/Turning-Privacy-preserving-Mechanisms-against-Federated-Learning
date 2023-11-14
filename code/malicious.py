import numpy as np


class Attack(object):
    def __init__(self, num_std):
        self.num_std = num_std
        self.grads_mean = None
        self.grads_stdev = None

    def attack(self, users_grads):
        new_grads = []
        if len(users_grads) == 0:
            return new_grads

        self.grads_mean = np.mean(users_grads, axis=0)
        self.grads_stdev = np.var(users_grads, axis=0) ** 0.5

        if self.num_std == 0:
            return

        mal_grads = self._attack_grads(self.grads_mean, self.grads_stdev)

        for _ in users_grads:
            new_grads.append(mal_grads)

        return new_grads

class DriftAttack(Attack):
    def __init__(self, num_std):
        super(DriftAttack, self).__init__(num_std)

    def _attack_grads(self, grads_mean, grads_stdev):
        grads_mean[:] -= self.num_std * grads_stdev[:]
        return grads_mean
