import logging
from tqdm import tqdm
import numpy as np
import random

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader
from dgl import load_graphs

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_graph_classification_dataset
from graphmae.models import build_model, build_classify_model


def graph_classification_evaluation(model, pooler, dataloader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_g, labels) in enumerate(dataloader):
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            out = model.embed(batch_g, feat)
            out = pooler(batch_g, out)

            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    return test_f1

## shiping
def getCallGraph(model, pooler, graphs, device):
    call_graphs = []
    # labels = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(graphs))):
            (address, pdgs, label) = graphs[i]
            # labels.append(label.unsqueeze(0))
            batch_g = dgl.batch(pdgs).to(device)
            feat = batch_g.ndata["tac_op"]
            out = model.embed(batch_g, feat)
            out = pooler(batch_g, out)
            srcs = []
            dsts = []
            with open("/home/huangshiping/study/gigahorse-toolchain/CallGraphs/" + address + ".txt", "r") as f:
                for line in f.readlines():
                    [src, dst] = line.strip().split("\t")
                    srcs.append(int(src))
                    dsts.append(int(dst))
            src_ids = torch.tensor(srcs, dtype=torch.int64)
            dst_ids = torch.tensor(dsts, dtype=torch.int64)
            call_graph = dgl.graph((src_ids, dst_ids), num_nodes = len(pdgs))
            call_graph.ndata['x'] = out.cpu()
            call_graphs.append([call_graph, label.unsqueeze(0)])
    return call_graphs



def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob=True, logger=None):
    train_loader, eval_loader = dataloaders

    epoch_iter = tqdm(iterable = range(max_epoch), position=0)
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g = batch
            batch_g = batch_g.to(device)

            feat = batch_g.ndata["tac_op"]
            model.train()
            loss, loss_dict = model(batch_g, feat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)
        if scheduler is not None:
            scheduler.step()
        # epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
        print(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    return model

            
def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    # graphs = [x[0] for x in batch]
    # labels = [x[1] for x in batch]
    batch_g = dgl.batch(batch)
    # labels = torch.cat(labels, dim=0)
    return batch_g

def callGraphFn(batch):
    graphs = [x[0].add_self_loop() for x in batch]
    # graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    graphs = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return graphs, labels

def classify(model, dataloader, optimizer, loss_fn, device):
    (train_loader, test_loader) = dataloader
    for epoch in range(400):
        model.train()
        loss_list = []
        for batch, labels in train_loader:
            batch_g = batch.to(device)
            y = labels.to(device)

            feat = batch_g.ndata['x']
            model.train()
            out = model(batch_g, feat)
            loss = loss_fn(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        print(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
        train_acc, train_f1, train_pre, train_rec = test(model, train_loader, device)
        print(f"train_acc: {train_acc:.4f}, train_f1: {train_f1:.4f}, train_pre: {train_pre:.4f}, train_rec: {train_rec:.4f}")
        test_acc, test_f1, test_pre, test_rec = test(model, test_loader, device)
        print(f"test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}, test_pre: {test_pre:.4f}, test_rec: {test_rec:.4f}")

def test(model, dataloader, device):
    y_pred = torch.tensor([], dtype=torch.int64)
    y_true = torch.tensor([], dtype=torch.int64)
    model.eval()
    for batch, labels in dataloader:
        g = batch.to(device)
        x = g.ndata['x']
        out = model(g, x)
        out = torch.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        y_pred = torch.cat([y_pred, pred.cpu()])
        y_true = torch.cat([y_true, labels.cpu()])
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro')
    return accuracy, f1, precision, recall

def main(args):
    # device = args.device if args.device >= 0 else "cpu"
    device = 1
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    pooling = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size

    # graphs, (num_features, num_classes) = load_graph_classification_dataset(dataset_name, deg4feat=deg4feat)
    args.num_features = num_hidden
    num_classes = 2

    ## Shiping
    # graphs = []
    address_labels = {}
    with open("/home/huangshiping/study/gigahorse-toolchain/address_labels_gigahorse.txt", "r") as f:
        for line in f.readlines():
            [address, label] = line.strip().split("\t")
            address_labels[address] = torch.tensor(int(label), dtype=torch.int64)
    addresses = list(address_labels.keys())
    # with open("/home/huangshiping/study/gigahorse-toolchain/address_labels_gigahorse.txt", "r") as f:
    #     lines = f.readlines()
    #     for i in tqdm(range(len(lines))):
    #         [address, label] = lines[i].strip().split("\t")
    #         pdgs, _ = load_graphs("/home/huangshiping/study/gigahorse-toolchain/PDGs/" + address + ".txt")
    #         graphs.append([pdgs, torch.tensor(int(label), dtype=torch.int64)])
    # random.seed(1234)
    # pretrain_index = random.sample(range(len(addresses)), 5000)
    pretrain_graphs = []
    graphs = []
    # for graph in graphs:
    for i in tqdm(range(len(addresses))):
        address = addresses[i]
        pdgs, _ = load_graphs("/home/huangshiping/study/gigahorse-toolchain/PDGs/" + address + ".txt")
        graphs.append((address, pdgs, address_labels[address]))
        pretrain_graphs.extend(pdgs)
    print(len(pretrain_graphs))
    train_loader = GraphDataLoader(pretrain_graphs, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    # train_idx = torch.arange(len(graphs))
    # train_sampler = SubsetRandomSampler(train_idx)
    
    # train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    # eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError

    acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        if not load_model:
            model = pretrain(model, pooler, (train_loader, ""), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob,  logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model.to(device)
        call_graphs = getCallGraph(model, pooler, graphs, device)
        # train_indexs = random.sample(range(len(call_graphs)), 3000)
        train_split = int(len(call_graphs) * 0.8)
        test_split = len(call_graphs) - train_split
        train_graphs, test_graphs = random_split(call_graphs, [train_split, test_split], generator=torch.Generator().manual_seed(42))
        # print(len(train_graphs))
        # print(len(test_graphs))
        train_loader = GraphDataLoader(train_graphs, collate_fn=callGraphFn, batch_size=batch_size, shuffle=True)
        test_loader = GraphDataLoader(test_graphs, collate_fn=callGraphFn, batch_size=batch_size)
        classify_model = build_classify_model(256, 256, 256, 2)
        classify_model.to(device)
        classify_optimizer = torch.optim.Adam(classify_model.parameters(), lr = 0.0001)
        classify_loss_fn = torch.nn.CrossEntropyLoss()
        classify_loss_fn.to(device)
        classify(classify_model, (train_loader, test_loader), classify_optimizer, classify_loss_fn, device)
        # print(labels.sum())
        # model.eval()
        # test_f1 = graph_classification_evaluation(model, pooler, eval_loader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False)
        # acc_list.append(test_f1)

    # final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    # print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    args.encoder = 'gin'
    args.decoder = 'mlp'
    args.device = 1
    args.batch_size = 128
    args.lr = 0.001
    args.save_model = False
    args.load_model = True
    args.pooling = "mean"
    print(args)
    main(args)
