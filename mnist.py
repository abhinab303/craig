from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import numpy as np
import time
from keras.regularizers import l2
import util
import pdb
import argparse
from pathlib import Path
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(description='MNIST')
parser.add_argument('--tau', type=float, default=1.0, help='tau for average', dest="tau")
parser.add_argument('--jb', type=str, help='job_name_for_file', dest="job_name")
parser.add_argument('--bj', type=str, help='base job_name_for_ folder', dest="base_job")
parser.add_argument('--lam', type=float, help='lambda for average', dest="lam")
args = parser.parse_args()

TRAIN_NUM = 60000

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

num_classes, smtk = 10, 0
Y_train_nocat = Y_train
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

batch_size = 32
# subset, random = False, False  # all
subset, random = True, False  # greedy
# subset, random = True, True  # random

EL2N_only = False
EL2N_plus_CRAIG = True

subset_size = .5 if subset else 1.0
epochs = 30
reg = 1e-4
runs = 5
save_subset = False

TAU = args.tau
LAMBDA = args.lam
BASE_JOB = args.base_job
JOB_NAME = args.job_name

base_dir = f"./tmp/{BASE_JOB}"
path = Path(base_dir)
path.mkdir(parents=True, exist_ok=True)
folder = f'{base_dir}/{JOB_NAME}'





train_loss, test_loss = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_acc, test_acc = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_time = np.zeros((runs, epochs))
grd_time, sim_time, pred_time = np.zeros((runs, epochs)), np.zeros((runs, epochs)), np.zeros((runs, epochs))
not_selected = np.zeros((runs, epochs))
times_selected = np.zeros((runs, len(X_train)))
number_of_iteration = np.zeros((runs, epochs))
el2n_list = [[]]*runs
best_acc = 0
print(f'----------- smtk: {smtk} ------------')

## TODO:
# note number of iteration
# implement EL2N score
# merge EL2N with Craig 

# can I make a function for it?



B = int(subset_size * TRAIN_NUM)
if save_subset:
    B = int(subset_size * len(X_train))
    selected_ndx = np.zeros((runs, epochs, B))
    selected_wgt = np.zeros((runs, epochs, B))

for run in range(runs):
    X_subset = X_train
    Y_subset = Y_train
    W_subset = np.ones(len(X_subset))
    ordering_time,similarity_time, pre_time = 0, 0, 0
    loss_vec, acc_vec, time_vec = [], [], []

    iteration_count = 0
    last_score = None

    model = Sequential()
    model.add(Dense(100, input_dim=784, kernel_regularizer=l2(reg)))
    model.add(Activation('sigmoid'))
    model.add(Dense(10, kernel_regularizer=l2(reg)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')

    for epoch in range(0, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        num_batches = int(np.ceil(X_subset.shape[0] / float(batch_size)))

        X_subset, Y_subset = shuffle(X_subset, Y_subset)

        for index in range(num_batches):
            X_batch = X_subset[index * batch_size:(index + 1) * batch_size]
            Y_batch = Y_subset[index * batch_size:(index + 1) * batch_size]
            W_batch = W_subset[index * batch_size:(index + 1) * batch_size]

            start = time.time()
            history = model.train_on_batch(X_batch, Y_batch, sample_weight=W_batch)
            train_time[run][epoch] += time.time() - start

            iteration_count += 1


        if subset:
            if random:
                # indices = np.random.randint(0, len(X_train), int(subset_size * len(X_train)))
                indices = np.arange(0, len(X_train))
                np.random.shuffle(indices)
                indices = indices[:int(subset_size * len(X_train))]
                W_subset = np.ones(len(indices))
            else:
                start = time.time()
                _logits = model.predict_proba(X_train)
                pre_time = time.time() - start
                features = _logits - Y_train

                el2n_cur = np.linalg.norm(features, axis=1)

                # create the average of the scores:
                if epoch < 1:
                    el2n = el2n_cur
                else:
                    el2n = TAU * el2n_cur + (1 - TAU) * last_score
                last_score = el2n

                if EL2N_only:
                    labels = np.argmax(Y_train, axis=1)
                    classes = np.unique(labels)
                    C = len(classes)  # number of classes
                    num_per_class = np.int32(np.ceil(np.divide([sum(labels == i) for i in classes], TRAIN_NUM) * B))
                    ordering_time = 0
                    similarity_time = 0
                    all_weight = np.ones(TRAIN_NUM)

                    per_class_subset = []
                    for c in classes:
                        B_cur = num_per_class[c]
                        class_indices_all = np.where(labels == c)[0]
                        sort_score = np.argsort(el2n[class_indices_all])    # index of class_indices_all not the actual subset.
                        # warm start:
                        # if epoch < 30:
                        if False:
                            subset_class = np.random.choice(sort_score, size=B_cur, replace=False)
                            subset_class = class_indices_all[subset_class]    # convert the index to actual subset
                        else:
                            subset_class = sort_score[-B_cur:]      # choose the difficult samples
                            # subset_class = sort_score[:B_cur]         # choose the easier samples  
                            subset_class = class_indices_all[subset_class]
                        
                        per_class_subset.extend(list(subset_class))
                    indices = np.array(per_class_subset)
                    W_subset = all_weight[indices]
                elif EL2N_plus_CRAIG:

                    indices, W_subset, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                        int(subset_size * len(X_train)), features, 'euclidean', smtk, 0, False, Y_train_nocat)

                    labels = np.argmax(Y_train, axis=1)
                    classes = np.unique(labels)
                    C = len(classes)  # number of classes
                    num_per_class = np.int32(np.ceil(np.divide([sum(labels == i) for i in classes], TRAIN_NUM) * B))
                    ordering_time = 0
                    similarity_time = 0
                    all_weight = np.zeros(TRAIN_NUM)
                    all_weight[indices] = W_subset

                    per_class_subset = []
                    for c in classes:
                        B_cur = num_per_class[c]
                        class_indices_all = np.where(labels == c)[0]
                        class_indices_sub = np.intersect1d(class_indices_all, indices)
                        class_indices_rem = np.array(list(set(class_indices_all) - set(class_indices_sub)))
                        # sort_score_idx = np.argsort(el2n[class_indices_all])    # index of class_indices_all not the actual subset.
                        # sel_wt = np.argsort(all_weight[class_indices_sub])
                        # sel_wt_idx = np.argsort(all_weight[class_indices_all])

                        el2n_score = el2n[class_indices_all]
                        craig_score = all_weight[class_indices_all]

                        el2n_score_norm = el2n_score / el2n_score.sum()
                        craig_score_norm = craig_score / craig_score.sum()
                        t = epoch + 1
                        wt = np.exp(-t/LAMBDA)
                        score = craig_score_norm * wt + el2n_score_norm * (1 - wt)
                        # score = craig_score_norm

                        sel_score = np.argsort(score)     # higher score will be at the end          
                        
                        final_subset = list(class_indices_all[sel_score[-B_cur:]])
                        per_class_subset.extend(final_subset)

                        # pdb.set_trace()
                    # pdb.set_trace()
                    indices = np.array(per_class_subset)
                    W_subset = all_weight[indices]

                else:
                    indices, W_subset, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                        int(subset_size * len(X_train)), features, 'euclidean', smtk, 0, False, Y_train_nocat)

                W_subset = W_subset / np.sum(W_subset) * len(W_subset)  # todo

            if save_subset:
                selected_ndx[run, epoch], selected_wgt[run, epoch] = indices, W_subset

            el2n_list[run].append(el2n) 

            grd_time[run, epoch], sim_time[run, epoch], pred_time[run, epoch] = ordering_time, similarity_time, pre_time
            times_selected[run][indices] += 1
            not_selected[run, epoch] = np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100
        else:
            pred_time = 0
            indices = np.arange(len(X_train))

        X_subset = X_train[indices, :]
        Y_subset = Y_train[indices]

        start = time.time()
        score = model.evaluate(X_test, Y_test, verbose=1)
        eval_time = time.time()-start

        start = time.time()
        score_loss = model.evaluate(X_train, Y_train, verbose=1)
        print(f'eval time on training: {time.time()-start}')

        test_loss[run][epoch], test_acc[run][epoch] = score[0], score[1]
        train_loss[run][epoch], train_acc[run][epoch] = score_loss[0], score_loss[1]
        best_acc = max(test_acc[run][epoch], best_acc)

        number_of_iteration[run][epoch] = iteration_count

        grd = 'random_wor' if random else 'grd_normw'
        print(f'run: {run}, {grd}, subset_size: {subset_size}, epoch: {epoch}, test_acc: {test_acc[run][epoch]}, '
              f'loss: {train_loss[run][epoch]}, best_prec1_gb: {best_acc}, not selected %:{not_selected[run][epoch]}')

    if save_subset:
        print(
            f'Saving the results to {folder}_{subset_size}_{grd}_{runs}')

        np.savez(f'{folder}_{subset_size}_{grd}_{runs}',
                 # f'_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}_subset',
                 train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                 train_time=train_time, grd_time=grd_time, sim_time=sim_time, pred_time=pred_time,
                 not_selected=not_selected, times_selected=times_selected,
                 subset=selected_ndx, weights=selected_wgt)
    else:
        print(
            f'Saving the results to {folder}_{subset_size}_{grd}_{runs}')

        np.savez(f'{folder}_{subset_size}_{grd}_{runs}',
                 # f'_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}',
                 train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                 train_time=train_time, grd_time=grd_time, sim_time=sim_time, pred_time=pred_time,
                 not_selected=not_selected, times_selected=times_selected)

