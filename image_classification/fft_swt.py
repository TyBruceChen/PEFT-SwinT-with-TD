import torch as th
from tqdm import tqdm
import numpy  as np
from timm import create_model
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from avalanche.evaluation.metrics.accuracy import Accuracy
from vtab import *
from vtab_config import config
import random
import os
import sys
import wandb
from timm.scheduler import CosineLRScheduler

best_acc = 0.0

def train(args, model, dl, tdl, opt, sched, epochs, critirion= th.nn.CrossEntropyLoss()):
    model.train()
    model = model.cuda()
    acc = 0.
    global old_name
    old_name = None
    for epoch in (pbar:=tqdm(range(epochs))):
        if log:
            logger.log({"epoch":epoch})
        for batch in dl:
            x, y = batch[0].cuda(), batch[1].cuda()
            opt.zero_grad()
            out = model(x)
            loss = critirion(out, y)
            loss.backward()
            opt.step()
            train_acc = torch.mean((out.argmax(dim=1).view(-1) == y).float()).item()
            pbar.set_description(f"e: {epoch}, l: {round(loss.item(), 7)}, train_acc: {round(train_acc,3)}, a:{round(acc,4)}")
            if sched is not None:
                sched.step(epoch)
            if log:
                logger.log({"loss":loss.item()})
                logger.log({"train_acc'": round(train_acc,3)})
        if epoch % 1 == 0 and epoch != 0:
            if epoch >= 50:
                sched = None
            acc = test(model, tdl)
            if acc > args.best_acc:
                args.best_acc = acc
                if old_name is not None:
                    os.remove(old_name)
                old_name = f"./swt_headT_{round(acc, 5)}_seed_{args.seed}.pt"
                th.save(swt.state_dict(), old_name)
            if log:
                logger.log({"val_acc": acc})
    model = model.cpu()
    return model


@th.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y)
    return acc.result()


def _parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dim",
        default=32,
        type=int,
        help="Number of trainable ranks."
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--dataset",
        default="cifar",
        type=str,
        choices=["cifar", "caltech101", "clevr_count", "clevr_dist", "diabetic_retinopathy",
                 "dmlab", "dsprites_loc", "dtd", "eurosat", "kitti", "oxford_flowers102",
                 "oxford_iiit_pet", "patch_camelyon", "resisc45", "smallnorb_azi",
                 "smallnorb_ele", "sun397", "svhn", "dsprites_ori"],
        help="Dataset to train"
    )
    parser.add_argument(
        "--evaluate",
        default=None,
        type=str,
        help="Evalute model only"
    )
    parser.add_argument('--model', type=str, default='swin_tiny_patch4_window7_224')

    parser.add_argument(
        "--check_point_path_swt",
        default=None,
        type=str,
        help="Pretrained model checkpoint path"
    )

    return parser.parse_args()


def main(sd = None):
    dataset_name = 'caltech101'
    num_classes=101
    batch_size = 128
    global logger, log

    args = _parse_args()
    print(args)
    name = args.dataset

    data_config = config[name]
    if sd is None:
        seed = data_config["seed"]
    else:
        seed = sd

    log = data_config["logger"]
    args.best_acc = 0.0
    args.seed = seed

    print(f"\n\nSeed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


    if log:
        run_name = f"LR_{args.lr}-fft-{dataset_name}"
        logger = wandb.init(project="CaRa-swt", name=run_name)
        logger.config.update(args)

    #from dataset import get_data_wafer
    #train_dl, val_dl, test_dl, _ = get_data_wafer('../../wafer_train/data/wafermap_train.npy')
    #from COVID19 import get_data_covid
    #train_dl, val_dl, test_dl, _ = get_data_covid('/home/tianyi_chen/.cache/kagglehub/datasets/tawsifurrahman/covid19-radiography-database/versions/5/COVID-19_Radiography_Dataset/')
    #num_classes = 5
    from get_torch_dataset import get_torch_dataset
    trainset, testset = get_torch_dataset(dataset_name)
    train_dl = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    global swt
    swt = create_model(args.model, checkpoint_path= args.check_point_path_swt, num_classes=num_classes, pretrained = True)

    trainable = []

    #swt.reset_classifier(num_classes)
    for n, p in swt.named_parameters():
        trainable.append(p)
        #total_param += p.numel()
    if args.evaluate is not None:
        print("Only evaluation")
        swt.load_state_dict(th.load(args.evaluate))
        acc = test(swt, test_dl)
        print(f"Accuracy: {acc}")
        sys.exit(0)

    optimizer = th.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05)
    scheduler = None
    #scheduler = CosineLRScheduler(optimizer, t_initial=100, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
    print(f'SW block weight: {swt.layers[2].blocks[0].attn.qkv.weight[1][-3:]}, header weight: {swt.head.fc.weight[-4:][-4:]}')
    swt = train(args, swt, train_dl, val_dl, optimizer, scheduler, epochs=100)
    print("\n\n Evaluating....")
    #_, _, test_dl, _ = get_data_wafer('../../wafer_train/data/wafermap_train.npy')
    acc = test(swt, val_dl)
    print(acc)
    if acc > args.best_acc:
        args.best_acc = acc
        os.remove(old_name)
        th.save(swt.state_dict(), f"./swt_{name}_{round(args.best_acc, 5)}_seed_{seed}.pt")

    print(f"Accuracy: {args.best_acc}")


if __name__ == "__main__":
    main()