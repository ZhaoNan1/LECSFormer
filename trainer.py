
import logging
from operator import mod
import os
import sys
from statistics import mean, mode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import summary
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.crack_datasets import *


def trainer(args, model, config):
    logging.basicConfig(filename=args.output_dirs + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # base_lr = args.base_lr

    train_data = Crack_Datasets(data_root=args.root_path,
                                img_list=os.path.join(args.root_path,'train.txt'),
                                img_size=args.img_size,
                                mode='train'
                                )
    val_data = Crack_Datasets(data_root=args.root_path,
                              img_list=os.path.join(args.root_path,'val.txt'),
                              img_size=args.img_size,
                              mode='val'
                              )
    train_loader = DataLoader(train_data,
                            batch_size=args.batch_size,
                            drop_last=True,
                            shuffle=True,
                            num_workers=12,
                            pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(),betas=config.TRAIN.OPTIMIZER.BETAS,eps=config.TRAIN.OPTIMIZER.EPS,
                            lr=config.TRAIN.BASE_LR,weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9,last_epoch=-1)
    iter_num = 0
    best_loss = float('inf')
    max_iterations = args.max_epochs * len(train_loader)
    model.train()
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    for epoch_num in tqdm(range(args.max_epochs),ncols=70):
        for i_batch, sample_batch in enumerate(train_loader):
            images, labels = sample_batch['image'], sample_batch['label']
            images, labels = images.cuda(), labels.cuda()

            output, mid_features = model(images)

            output_loss_ce = ce_loss(output,labels) * 5
            midfeatures_loss_ce = sum([ce_loss(mid_features[i],labels) for i in range(len(mid_features))]) 
            loss = output_loss_ce + midfeatures_loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = config.TRAIN.BASE_LR * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1
            logging.info('iteration %d : loss : %f, mid_ce_loss: %f, loss_ce: %f' % (iter_num,
                                                                                   loss.item(),
                                                                                   midfeatures_loss_ce.item(),
                                                                                   output_loss_ce.item()))
        
        if (epoch_num + 1) % 5 == 0:
            model.eval()
            loss_val = []
            for i_batch, sample_batch in tqdm(enumerate(val_loader)):
                image, label = sample_batch['image'], sample_batch['label']
                image, label = image.cuda(), label.cuda()
                output, mid_features = model(image)
                loss_ = ce_loss(output, label)
                loss_val.append(loss_.item())
            mean_loss_val = mean(loss_val)

        if (epoch_num + 1) % 10 == 0:
            save_model_path = os.path.join(args.output_dirs,'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(),save_model_path)
            logging.info("save model to {}".format(save_model_path))

        if epoch_num >= args.max_epochs - 1:
            save_model_path = os.path.join(args.output_dirs,'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(),save_model_path)
            logging.info("save model to {}".format(save_model_path))
    return
                



    
    
