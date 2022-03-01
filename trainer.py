import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
import time
import tifffile
import tqdm
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import lovasz_softmax
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from datasets.dataset_synapse import Synapse_dataset


def cal(pred, label):

    NUM_CATEGORIES=5
    tp = np.zeros(NUM_CATEGORIES)
    fp = np.zeros(NUM_CATEGORIES)
    fn = np.zeros(NUM_CATEGORIES)
    out = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0)

    prediction = out.cpu().detach().numpy()

    label = label.cpu()
    label = label.numpy()
    for cat in range(NUM_CATEGORIES):
        tp[cat] += ((prediction == cat) & (label == cat) & (label < NUM_CATEGORIES)).sum()
        fp[cat] += ((prediction == cat) & (label != cat) & (label < NUM_CATEGORIES)).sum()
        fn[cat] += ((prediction != cat) & (label == cat) & (label < NUM_CATEGORIES)).sum()

    np.seterr(divide='ignore', invalid='ignore')
    iou = np.divide(tp, tp + fp + fn)

    m=iou.mean()
    return m

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # fen hao zu le 758iterations per epoch 758ge batch
    #if args.n_gpu > 1:
    #    model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()

    dice_loss = DiceLoss(num_classes)
    # lovasz_loss = lovasz_softmax(num_classes)#lovasz_softmax_flat(probas, labels,lovasz_softmax
    #optimizer = optim.Adam()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)  # yohua SGD
    # scheduler= optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,10], gamma=0.1, last_epoch=-1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120], gamma=0.5, last_epoch=-1)
    # optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))
    writer = SummaryWriter(snapshot_path + '/log')
    '''
    snapshot ='/root/data/Try/Try3-62/networks/model/TU_02_Vai_wu_256256/TU_R50-ViT-B_16_skip3_epo150_bs8_256/epoch_79.pth'
    checkpoint = torch.load(snapshot, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = 80
    print('load epoch{} succeed!'.format(start_epoch))
    '''
    # print(snapshot_path)
    iter_num = 0
    max_epoch = args.max_epochs

    max_iterations = args.max_epochs * len(trainloader)
    # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # print('epoch:',epoch_num)
        epoch_loss = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            # print('iter_num',iter_num)
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            image_batch = image_batch.cpu()
            image_batch = image_batch.numpy()
            image_batch = image_batch.transpose([0, 3, 1, 2])

            # image_batch torch.Size([12, 3, 256, 256])
            image_batch = torch.from_numpy(image_batch)
            image_batch = image_batch.cuda()
            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            # lovasz_loss=lovasz_softmax(outputs, label_batch)##lovasz_softmax_flat(probas, labels,
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = loss_ce+ 1.5 * loss_dice  # loss
            #print('9999999999999',outputs.shape)
            epoch_loss = epoch_loss + loss
            miou=cal(outputs,label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            #lr = optimizer.param_groups[0]['lr']
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            #writer.add_scalar('info/miou', miou, iter_num)

            logging.info('iteration %d : lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            #logging.info('iteration %d : lr : %f, loss : %f, loss_ce: %f' % (iter_num, lr_, loss.item(), loss_ce.item()))
            if iter_num % 1 == 0:
                image = image_batch[0, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #scheduler.step()
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_num,
        }
        torch.save(checkpoint, save_mode_path)
        logging.info("save model to {}".format(save_mode_path))


    writer.close()
    return "Training Finished!"