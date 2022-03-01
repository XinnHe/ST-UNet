import argparse
import logging

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchsummary import summary
from torchvision.models import resnet50
from thop import profile


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/private/data/Vai256_npz/test_npz', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Vai', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()


    NUM_CATEGORIES=5
    tp = np.zeros(NUM_CATEGORIES)
    fp = np.zeros(NUM_CATEGORIES)
    fn = np.zeros(NUM_CATEGORIES)
    tn = np.zeros(NUM_CATEGORIES)
    Time = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):

        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        image_batch = image_batch.cpu()
        image_batch = image_batch.numpy()
        image_batch = image_batch.transpose([0, 3, 1, 2])
        # print('image_batch', type(image_batch))
        # print('label_batch', label_batch.shape)
        image_batch = torch.from_numpy(image_batch)
        image_batch = image_batch.cuda()
        image, label, case_name = image_batch, label_batch , sampled_batch['case_name'][0]
        prediction, time = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        Time = time + Time
        label_batch = label_batch.cpu()
        label = label_batch.numpy()
        for cat in range(NUM_CATEGORIES):
            #tp[cat] += ((prediction == cat) & (label == cat) & (label < NUM_CATEGORIES)).sum()
            #fp[cat] += ((prediction == cat) & (label != cat) & (label < NUM_CATEGORIES)).sum()
            #fn[cat] += ((prediction != cat) & (label == cat) & (label < NUM_CATEGORIES)).sum()

            tp[cat] += ((prediction == cat) & (label == cat) & (label < NUM_CATEGORIES)).sum()
            fp[cat] += ((prediction == cat) & (label != cat) & (label < NUM_CATEGORIES)).sum()
            fn[cat] += ((prediction != cat) & (label == cat) & (label < NUM_CATEGORIES)).sum()
            tn[cat] += ((prediction != cat) & (label != cat) & (label < NUM_CATEGORIES)).sum()
        # accumulate statistics for IOU-3

        # compute IOU-3
    nfiles = len(testloader)
    print('Generated segmentations in %s ms/per -- %s FPS' % (Time / nfiles * 1000, nfiles / Time))
    print('Generated segmentations in %s seconds' % (Time))
    np.seterr(divide='ignore', invalid='ignore')
    iou = np.divide(tp, tp + fp + fn)

    pre = np.divide(tp, (tp + fp))
    recall = np.divide(tp, (tp + fn))
    f1 = np.divide(2 * pre * recall, (pre + recall))
    acc = np.divide((tp+tn).sum(), (tp + fn+fp+tn).sum())
    print('---------------------------------------------------')
    print('IOU:  ', iou)
    print('mIOU: ', iou.mean())
    print('---------------------------------------------------')
    # print('recall',recall)
    print('F1', f1)
    print('Ave.F1', f1.mean())
    print('Acc', acc)
    print('---------------------------------------------------')

    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.img_size = 256
    args.batch_size = 8
    dataset_name = 'Vai_256'#'PotsNo256'#
    dataset_config = {
        'Vai_256': {
            'Dataset': Synapse_dataset,
            'volume_path': '/private/hexin/data/Vai_256_npz/test_npz',
            'list_dir': './lists/lists_Vai_256',
            'num_classes': 6,
            'z_spacing': 1,
        },
        'Pots_256': {
            'Dataset': Synapse_dataset,
            'volume_path': '/private/hexin/data/Potsdam/Pots_256_npz/test_npz/',
            'list_dir': './lists/lists_Pots_256',
            'num_classes': 6,
            'z_spacing': 1,
        },
    }
    #dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = False#True

    # name the same snapshot defined in train script!
    args.exp = 'STUNet_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    print('-------------------------------------------')
    print(snapshot_path)
    print('------------------------------------------')

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip #3
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size) #16
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    #summary(net, input_size=(3, 256, 256), batch_size=-1)



    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in net.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')


    snapshot='/private/hexin/data/Try/ST-UNet/networks/CSTmodel/TU_10_Vai_wu_256256/TU_R50-ViT-B_16_skip3_epo150_bs8_256/epoch_99.pth'

    checkpoint = torch.load(snapshot, map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
    #if torch.cuda.device_count() > 1:
        # 如果有多个GPU，将模型并行化，用DataParallel来操作。这个过程会将key值加一个"module. ***"。gg
    #    net = nn.DataParallel(net)
    net.load_state_dict(checkpoint['model'])


    snapshot_name = snapshot_path.split('/')[-1]

    #log_folder = './test_log/test_log_' + args.exp
    #os.makedirs(log_folder, exist_ok=True)
    #logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)
    args.is_savenii=True
    if args.is_savenii:
        args.test_save_dir = '/private/hexin/data/Try/ST-UNet/test/test_'+args.exp+'/'
        test_save_path=args.test_save_dir
        #test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        print(test_save_path)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
#/private/data/TransUNet-main/networks/model/TU_Vai256/TU_R50-ViT-B_16_skip3_epo150_bs12_256/epoch_149.pth

