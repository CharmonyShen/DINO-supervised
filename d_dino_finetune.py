import os
import argparse
import json
from pathlib import Path
import datetime
import time

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchmetrics import Precision, Recall, F1Score,ConfusionMatrix

import utils
import vision_transformer as vits

def land_use_classify(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        pretrain_model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=args.num_labels)
        embed_dim = pretrain_model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        pretrain_model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=args.num_labels)
        embed_dim = pretrain_model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        # pretrain_model = torchvision_models.__dict__[args.arch](num_classes=768)

        if 'swin' in args.arch:
            pretrain_model = torchvision_models.__dict__[args.arch](num_classes=args.num_labels)
            embed_dim = pretrain_model.head.weight.shape[1]
        else:
            pretrain_model = torchvision_models.__dict__[args.arch](num_classes=args.num_labels)
            embed_dim = pretrain_model.fc.weight.shape[1]
        # pretrain_model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    pretrain_model.cuda()
    # pretrain_model.add_module('Classifier',nn.Linear(1000, args.num_labels))
    # pretrain_model.fc.add_module('classifier',nn.Linear(embed_dim, args.num_labels))
    print(pretrain_model)

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "test"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # load weights to evaluate
    if args.evaluate:
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        # print(state_dict.keys())
        state_dict = state_dict['state_dict']
        msg = pretrain_model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
        test_stats,con_matr = validate_network(val_loader, pretrain_model, args.n_last_blocks, args.avgpool_patchtokens, args)
        print(con_matr)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    utils.load_pretrained_weights(pretrain_model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Pretrained Model {args.arch} built.")

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=train_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
    # set optimizer
    optimizer = torch.optim.SGD(
        pretrain_model.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "luc_checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=pretrain_model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    
    start_time = time.time()
    print("Starting finetuning!")
    print('start_epoch=',start_epoch, 'args.epochs=',args.epochs)
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(pretrain_model, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats,con_matr = validate_network(val_loader, pretrain_model, args.n_last_blocks, args.avgpool_patchtokens, args)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if best_acc <= test_stats["acc1"]:
                print(con_matr)
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": pretrain_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            if epoch > 15:
                if best_acc <= test_stats["acc1"]:
                    torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_best.pth.tar"))
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Training of the supervised linear classifier on remote sensing images completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(pretrain_model, optimizer, loader, epoch, n, avgpool):
    pretrain_model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # print(inp.shape)
        # forward
        # with torch.no_grad():
        # if "vit" in args.arch:
        #     intermediate_output = pretrain_model.get_intermediate_layers(inp, n)
        #     output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        #     if avgpool:
        #         output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
        #         output = output.reshape(output.shape[0], -1)
        # else:
        output = pretrain_model(inp)
        # print(output.shape)
        # print(linear_classifier)
        # output = pretrain_model(inp)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)
        # print('shape:',output.shape,target.shape)
        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        # if predicts != '':
        #     predicts = torch.cat((predicts, output), 0)
        #     labels = torch.cat((labels, target), 0)
        # else:
        #     predicts = output
        #     labels = target
        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # print('pred,label',predicts.shape, labels.shape)
    # prec = precision_func(predicts, labels)
    # rec = recall_func(predicts, labels)
    # f1 = f1_func(predicts, labels)
    # # print(len(predicts),len(labels))
    # print('f1:',f1,'prec:',prec,'rec:',rec)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def validate_network(val_loader, pretrain_model, n, avgpool, args):
    pretrain_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    precision_func = Precision(num_classes=args.num_labels, average='macro').cuda()
    recall_func = Recall(num_classes=args.num_labels, average='macro').cuda()
    f1_func = F1Score(num_classes=args.num_labels, average='macro').cuda()
    confmat = ConfusionMatrix(num_classes=args.num_labels).cuda()
    
    confusion_matrix = ''
    predicts = ''
    labels = ''
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = pretrain_model(inp)
        loss = nn.CrossEntropyLoss()(output, target)
        # print('predicts',predicts)
        # print('output',output)
        if predicts != '':
            predicts = torch.cat((predicts, output), 0)
            labels = torch.cat((labels, target), 0)
        else:
            predicts = output
            labels = target
        acc1, = utils.accuracy(output, target, topk=(1,))
        # print('output',output)
        # prec = precision_func(output, target)
        # rec = recall_func(output, target)
        # f1 = f1_func(output, target)
        if confusion_matrix == '':
            confusion_matrix = confmat(output, target)
        else:
            confusion_matrix = confusion_matrix + confmat(output, target)
        
        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['prec'].update(prec.item(), n=batch_size)
        # metric_logger.meters['rec'].update(rec.item(), n=batch_size)
        # metric_logger.meters['f1'].update(f1.item(), n=batch_size)

        # if pretrain_model.module.num_labels >= 5:
        #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # if pretrain_model.module.num_labels >= 5:
        # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        #   .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # else:
    prec = precision_func(predicts, labels)
    rec = recall_func(predicts, labels)
    f1 = f1_func(predicts, labels)
    # print(len(predicts),len(labels))
    print('f1:',f1,'prec:',prec,'rec:',rec)
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    # print(confusion_matrix)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, confusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=7, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    land_use_classify(args)
