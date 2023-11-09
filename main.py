import opt
from utils import normalize_adj, sparse_mx_to_torch_sparse_tensor, generate_split, compute_semk_hop_new
from datafile import get_dataset
from torch_geometric.utils import remove_self_loops, add_self_loops
import os
from model_khop import SemGcnGat
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from metrics import masked_softmax_cross_entropy, evaluation
from simple_param.sp import SimpleParam
import scipy.sparse
from torch_sparse import SparseTensor
import time
import scipy.sparse as sparse

default_param = {
    'learning_rate': 0.01,
    'weight_decay': 5e-4,
    'dropout': 0.5
}

os.environ["CUDA_VISIBLE_DEVICES"] = opt.args.gpuid
# device = torch.device("cuda:{}".format(opt.args.gpuid))
device = torch.device("cuda")
# device = torch.device("cpu")
print('device:{}'.format(device))
path = os.path.expanduser('./datasets')
path = os.path.join(path, opt.args.name)

dataset = get_dataset(path, opt.args.name)
if opt.args.name in ['Chameleon', 'arxiv-year']:
    data = dataset.data
    data = data.to(device)
else:
    data = dataset[0]
data = data.to(device)
if len(data.y.shape) > 1:
    labels = data.y.squeeze(dim=-1)
else:
    labels = data.y

features = data.x

# adj = init_edge(data)
edge_index = data.edge_index.cpu()
edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
edge_index, _ = remove_self_loops(edge_index, None)

row, col = edge_index
row = row - row.min()  # for sampling
values = np.ones(row.size(0), dtype=int)
adj = sparse.coo_matrix((values, (row, col)), shape=(data.num_nodes, data.num_nodes))

adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj_dig = adj
# adj = normalize_adj(adj)
adj_dig = normalize_adj(adj + sparse.eye(adj.shape[0]))
adj = sparse_mx_to_torch_sparse_tensor(adj)
adj_dig = sparse_mx_to_torch_sparse_tensor(adj_dig)
# adj = adj.to(device)
adj_dig = adj_dig.to(device)
train_percent, val_percent = opt.args.train_percent, opt.args.val_percent
print('train percent={}, valid percent={}'.format(train_percent, val_percent))

# parse param
sp = SimpleParam(default=default_param)
param_data = sp(source=opt.args.param, preprocess='nni')
ckpt_path = f'ckpt_{opt.args.name}_{opt.args.model}_{train_percent}_semk_hop'
if not Path(ckpt_path).exists():
    os.mkdir(ckpt_path)


for spid in range(1, 11):
    split_id = spid
    save_path = './splits/'
    filepath = os.path.join(save_path, '{}_split_{}_{}_{}.npz'
                            .format(opt.args.name, train_percent, val_percent, split_id))
    if not os.path.exists(filepath):
        train_mask, val_mask, test_mask = generate_split(data.num_nodes, train_ratio=train_percent, val_ratio=val_percent)
        np.savez(filepath, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    else:
        with np.load(filepath) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

    y_train = torch.LongTensor(np.zeros(labels.shape)).to(device)
    y_val = torch.LongTensor(np.zeros(labels.shape)).to(device)
    y_test = torch.LongTensor(np.zeros(labels.shape)).to(device)

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    y_train[train_mask] = labels.long()[train_mask]
    y_val[val_mask] = labels.long()[val_mask]
    y_test[test_mask] = labels.long()[test_mask]

    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)


    if not os.path.exists(f'saved/{opt.args.name}_{opt.args.sem_k}_hop.pt'):
        compute_semk_hop_new(features, adj, opt.args.name)
    print(f'saved/{opt.args.name}_{opt.args.sem_k}_hop.pt')
    adj_sem = torch.load(f'saved/{opt.args.name}_{opt.args.sem_k}_hop.pt', map_location=device)
    adj_sem = sparse_mx_to_torch_sparse_tensor(adj_sem).to(device)

    #checkpoint_test_acc_path = f'./{ckpt_path}/{opt.args.name}-{opt.args.model}-tr_{train_percent}-val_{val_percent}-split_{split_id}-eAtt_d_{opt.args.eAtt_d}-sem_k_{opt.args.sem_k}-tsf_lay_{opt.args.tsf_layer}-tsf_head_{opt.args.tsf_head}-tsf_drop_{opt.args.tsf_drop}-ly_{opt.args.gcn_layer}-ffn_d_{opt.args.ffn_dim}-hid_{opt.args.hidden_dim}-check_test_acc.pt'

    checkpoint_val_acc_path = f'./{ckpt_path}/{opt.args.name}-{opt.args.model}-tr_{train_percent}-val_{val_percent}-split_{split_id}-eAtt_d_{opt.args.eAtt_d}-sem_k_{opt.args.sem_k}-tsf_lay_{opt.args.tsf_layer}-tsf_head_{opt.args.tsf_head}-tsf_drop_{opt.args.tsf_drop}-ly_{opt.args.gcn_layer}-ffn_d_{opt.args.ffn_dim}-hid_{opt.args.hidden_dim}-check_val_acc.pt'

    seed = 1024
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    model = SemGcnGat(device=device, n_layers=opt.args.gcn_layer,
                      n_features=param_data['n_features'], hidden_dim=opt.args.hidden_dim,
                      dropout=param_data['dropout'],
                      n_classes=param_data['n_classes'],
                      args=opt.args, ffn_dim=opt.args.ffn_dim, tsf_layer=opt.args.tsf_layer,
                      tsf_head=opt.args.tsf_head, tsf_drop=opt.args.tsf_drop, eAtt_d=opt.args.eAtt_d)

    model.to(device)
    loss_func = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=param_data['learning_rate'],
                                 weight_decay=param_data['weight_decay'])
    best_test_acc = 0.0
    best_val_loss = 0.0
    best_epoch = 0
    best_val_acc = 0.0
    # train_accls = []
    # val_accls = []
    # test_accls = []


    # res=[]
    epoch_time = []
    for epoch in range(opt.args.n_epoch):
        t = time.time()

        # train
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_dig, adj_sem)
        loss = masked_softmax_cross_entropy(loss_func, output, y_train, train_mask)
        # train_acc = masked_accuracy(output, y_train, train_mask)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        # t_test = time.time()
        output = model(features, adj_dig, adj_sem)

        # train_acc = masked_accuracy(output, y_train, train_mask)

        train_evalue, train_kappa, train_mcc = evaluation(output[train_mask],
                                                          y_train[train_mask])
        train_acc = train_evalue['accuracy']
        train_precision = train_evalue['macro avg']['precision']
        train_recall = train_evalue['macro avg']['recall']
        train_f1_score = train_evalue['macro avg']['f1-score']

        # train_accls.append(train_acc)

        # val_acc = masked_accuracy(output, y_val, val_mask)
        val_evalue, val_kappa, val_mcc = evaluation(output[val_mask],
                                                    y_val[val_mask])
        val_acc = val_evalue['accuracy']
        val_precision = val_evalue['macro avg']['precision']
        val_recall = val_evalue['macro avg']['recall']
        val_f1_score = val_evalue['macro avg']['f1-score']
        # val_accls.append(val_acc)
        val_loss = masked_softmax_cross_entropy(loss_func, output, y_val,
                                                val_mask).item()
        epoch_time.append(time.time() - t)
        # test_acc = masked_accuracy(output, y_test, test_mask)
        test_evalue, test_kappa, test_mcc = evaluation(output[test_mask],
                                                       y_test[test_mask])
        test_acc = test_evalue['accuracy']
        test_precision = test_evalue['macro avg']['precision']
        test_recall = test_evalue['macro avg']['recall']
        test_f1_score = test_evalue['macro avg']['f1-score']

        print(
            "{} | epoch {} | gcn nhid:{} | gcn nlayer:{} | train ACC:{}% | val ACC:{}% |"
            " test_ACC:{}% |test_P:{} |test_R:{} |test_F1:{} | train loss:{} |"
            " val loss:{} | tsf_dim:{} | tsf_layer:{} |"
            " tsf_head:{} | tsf_drop:{} | sem_k:{} | eAtt_d:{} | split_idx:{} | ep_time:{}".format(
                opt.args.name,
                epoch, opt.args.hidden_dim, opt.args.gcn_layer,
                np.round(train_acc * 100, 4),
                np.round(val_acc * 100, 4), np.round(test_acc * 100, 4),
                np.round(test_precision, 4), np.round(test_recall, 4),
                np.round(test_f1_score, 4),
                np.round(train_loss, 4),
                np.round(val_loss, 4), opt.args.ffn_dim, opt.args.tsf_layer, opt.args.tsf_head, opt.args.tsf_drop,
                opt.args.sem_k, opt.args.eAtt_d, split_id, epoch_time[-1]))

        # early stopping
        """
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                [model.state_dict(), output, labels, epoch, train_acc, train_loss,
                 val_loss,
                 val_acc, test_acc, train_evalue, train_kappa, train_mcc,
                 val_evalue, val_kappa, val_mcc, test_evalue, test_kappa, test_mcc,
                 test_precision, test_recall, test_f1_score, epoch_time],
                checkpoint_test_acc_path)
        """
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                [model.state_dict(), output, labels, epoch, train_acc, train_loss,
                 val_loss,
                 val_acc, test_acc, train_evalue, train_kappa, train_mcc,
                 val_evalue, val_kappa, val_mcc, test_evalue, test_kappa, test_mcc,
                 test_precision, test_recall, test_f1_score, epoch_time],
                checkpoint_val_acc_path)
            curr_step_val_acc = 0
        else:
            curr_step_val_acc += 1
            if curr_step_val_acc >= opt.args.early_stopping and epoch > 1000:  # >
                print(
                    "=====================early_stopping===========================")
                break