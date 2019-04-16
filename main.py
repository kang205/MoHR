import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm

from sampler import WarpSampler
from model import MoHR

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help="filename of dataset (*.npy)")
parser.add_argument('--batch_size', type=int, default=10000, help="batch size of each interation (default:10000)")
parser.add_argument('--latent_dimension', type=int, default=10, help="latent dimensionality K (default:10)")
parser.add_argument('--learning_rate', type=float, default=0.01, help="learning rate (default:0.01)")
parser.add_argument('--maximum_epochs', type=int, default=2000, help="maximum training epochs (default:2000)")
parser.add_argument('--alpha', type=float, default=1.0, help="coefficient on task T_I (default:1.0)")
parser.add_argument('--beta', type=float, default=0.1, help="coefficient on task T_R (default:0.1)")
parser.add_argument('--norm', type=float, default=1.0, help="maximum length of vectors (default:1.0)")
parser.add_argument('--lambda_bias', type=float, default=1e-4, help="coefficient lambda on bias terms (default:1e-4)")
parser.add_argument('--gamma', type=float, default=0.5, help="proportion of long-term preferences (default:0.5)")

args = parser.parse_args()

dataset = np.load(args.dataset+'Partitioned.npy')

[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
f = open('MoHR_%s_%d_%g_%g_%g_%g_%g_%g.txt' % (
    args.dataset, args.latent_dimension, args.learning_rate, args.alpha, args.beta, args.norm, args.lambda_bias,
    args.gamma), 'w')

# count postive events
oneiteration = 0
user_train_set = dict()
for user in user_train:
    oneiteration += len(user_train[user])
    user_train_set[user] = set(user_train[user])

# optimizing for sampler
Relationships = set()
for item in Item:
    for cat in Item[item]['related']:
        Relationships.add(cat)
Relationships = list(Relationships)
num_rel = len(Relationships)
invRelationships = dict()
for i in range(len(Relationships)):
    invRelationships[Relationships[i]] = i

for item in Item:
    for r in Item[item]['related']:
        Item[item]['related'][r] = set(Item[item]['related'][r])

    item_i_mask_list = []
    item_i_mask = np.zeros([num_rel + 1])
    for r in Item[item]['related'].keys():
        if len(Item[item]['related'][r]) != 0:
            item_i_mask[invRelationships[r] + 1] = 1.0
            item_i_mask_list.append(invRelationships[r] + 1)
    item_i_mask[0] = 1.0
    item_i_mask_list.append(0)
    Item[item]['mask'] = item_i_mask
    Item[item]['mask_list'] = item_i_mask_list

# define sampler with multi-processing
sampler = WarpSampler(user_train, user_train_set, Item, usernum, itemnum, Relationships, batch_size=args.batch_size,
                      n_workers=4)
valid_sampler = WarpSampler(user_train, user_train_set, Item, usernum, itemnum, Relationships,
                            batch_size=args.batch_size, is_test=True, User_test=user_validation, n_workers=2)
test_sampler = WarpSampler(user_train, user_train_set, Item, usernum, itemnum, Relationships,
                           batch_size=args.batch_size, is_test=True, User_test=user_test, n_workers=2)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

model = MoHR(usernum, itemnum, Relationships, args)
sess.run(tf.initialize_all_variables())

best_valid_auc = 0.5
best_iter = 0
num_batch = oneiteration / args.batch_size


try:
    for i in range(args.maximum_epochs):

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            batch = sampler.next_batch()
            batch_u, batch_i, batch_i_mask, batch_ui_r, batch_ui_rp, batch_j, batch_jp, batch_lp_r, batch_lp_i, batch_lp_j, batch_lp_jp = batch

            _, train_loss, train_auc = sess.run((model.gds, model.loss, model.auc),
                                                {model.batch_u: batch_u,
                                                 model.batch_i: batch_i,
                                                 model.batch_ui_r: batch_ui_r,
                                                 model.batch_ui_rp: batch_ui_rp,
                                                 model.batch_j: batch_j,
                                                 model.batch_jp: batch_jp,
                                                 model.batch_lp_r: batch_lp_r,
                                                 model.batch_lp_i: batch_lp_i,
                                                 model.batch_lp_j: batch_lp_j,
                                                 model.batch_lp_jp: batch_lp_jp
                                                 }
                                                )

        if i % 10 == 0:
            f.write('#iter %d: loss %f, auc %f\n' % (i, train_loss, train_auc))
            _valid_auc = 0.0
            _test_auc = 0.0
            n_batch = 1000000 / args.batch_size

            for _ in range(n_batch):
                batch = valid_sampler.next_batch()
                batch_u, batch_i, batch_i_mask, batch_ui_r, batch_ui_rp, batch_j, batch_jp, batch_lp_r, batch_lp_i, batch_lp_j, batch_lp_jp = batch

                valid_loss, valid_auc = sess.run((model.loss, model.auc),
                                                 {model.batch_u: batch_u,
                                                  model.batch_i: batch_i,
                                                  model.batch_ui_r: batch_ui_r,
                                                  model.batch_ui_rp: batch_ui_rp,
                                                  model.batch_j: batch_j,
                                                  model.batch_jp: batch_jp,
                                                  model.batch_lp_r: batch_lp_r,
                                                  model.batch_lp_i: batch_lp_i,
                                                  model.batch_lp_j: batch_lp_j,
                                                  model.batch_lp_jp: batch_lp_jp,
                                                  }
                                                 )

                batch = test_sampler.next_batch()
                batch_u, batch_i, batch_i_mask, batch_ui_r, batch_ui_rp, batch_j, batch_jp, batch_lp_r, batch_lp_i, batch_lp_j, batch_lp_jp = batch
                test_loss, test_auc = sess.run((model.loss, model.auc),
                                               {model.batch_u: batch_u,
                                                model.batch_i: batch_i,
                                                model.batch_ui_r: batch_ui_r,
                                                model.batch_ui_rp: batch_ui_rp,
                                                model.batch_j: batch_j,
                                                model.batch_jp: batch_jp,
                                                model.batch_lp_r: batch_lp_r,
                                                model.batch_lp_i: batch_lp_i,
                                                model.batch_lp_j: batch_lp_j,
                                                model.batch_lp_jp: batch_lp_jp,
                                                }
                                               )

                _valid_auc += valid_auc
                _test_auc += test_auc
            _valid_auc /= n_batch
            _test_auc /= n_batch

            f.write('%f %f\n' % (_valid_auc, _test_auc))
            f.flush()
            if _valid_auc > best_valid_auc:
                best_valid_auc = _valid_auc
                best_test_auc = _test_auc
                best_iter = i
                model.save(sess)
            elif i >= best_iter + 50:
                break
except:
    f.close()
    sampler.close()
    valid_sampler.close()
    test_sampler.close()
    exit(1)
sampler.close()
valid_sampler.close()
test_sampler.close()

f.write('Finished! %f, %f\n' % (best_valid_auc, best_test_auc))
f.close()
