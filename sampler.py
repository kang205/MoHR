import numpy as np
import random
from multiprocessing import Process, Queue


def sample_function(User, user_train_set, Item, usernum, itemnum, Relationships, batch_size, result_queue, SEED,
                    is_test=False, User_test=[]):
    num_rel = len(Relationships)
    invRelationships = dict()
    for i in range(len(Relationships)):
        invRelationships[Relationships[i]] = i

    def sample_ui():
        if not is_test:
            user = np.random.randint(0, usernum)
            while len(User[user]) <= 1: user = np.random.randint(0, usernum)
        else:
            user = np.random.randint(0, usernum)
            while len(User_test[user]) <= 1: user = np.random.randint(0, usernum)
        num_item = len(User[user])
        # find postive item pair

        if not is_test:
            item_i = np.random.randint(0, num_item - 1)
            item_j = item_i + 1

            item_i = User[user][item_i]
            item_j = User[user][item_j]
        else:
            item_i = User_test[user][0]
            item_j = User_test[user][1]

        item_i_mask = Item[item_i]['mask']
        item_i_mask_list = Item[item_i]['mask_list']

        pr = []
        flag = True
        for r in Item[item_i]['related'].keys():
            if item_j in Item[item_i]['related'][r]:
                pr.append(invRelationships[r] + 1)
                flag = False
        if flag:
            pr.append(0)
        item_ui_r = random.sample(pr, 1)[0]

        while True:
            item_ui_rp = np.random.randint(0, num_rel + 1)  # np.random.choice(item_i_mask_list,1)[0]
            if not item_ui_rp in pr: break

        s = user_train_set[user]
        item_jp = np.random.randint(0, itemnum)
        while item_jp in s or item_jp == item_i or item_jp == item_j: item_jp = np.random.randint(0, itemnum)
        return user, item_i, item_i_mask, item_ui_r, item_ui_rp, item_j, item_jp

    def sample_ii():

        while True:
            i = np.random.randint(0, itemnum)
            while len(Item[i]['related']) == 0:
                i = np.random.randint(0, itemnum)
            r = invRelationships[np.random.choice(Item[i]['related'].keys(), 1)[0]]

            num_item = len(Item[i]['related'][Relationships[r]])
            if num_item != 0:
                break

        j = random.sample(Item[i]['related'][Relationships[r]], 1)[0]

        jp = np.random.randint(0, itemnum)
        while jp in Item[i]['related'][Relationships[r]] or jp == i: jp = np.random.randint(0, itemnum)
        return r, i, j, jp

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            batch1 = sample_ui()
            batch2 = sample_ii()

            one_batch.append(batch1 + batch2)
        result_queue.put(zip(*one_batch))


class WarpSampler(object):

    def __init__(self, User, user_train_set, Item, usernum, itemnum, Relationships, batch_size=10000, n_workers=2,
                 is_test=False, User_test=[]):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      user_train_set,
                                                      Item,
                                                      usernum,
                                                      itemnum,
                                                      Relationships,
                                                      batch_size,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      is_test,
                                                      User_test)))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
