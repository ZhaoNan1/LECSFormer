import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.crack_datasets import Crack_Datasets
from networks.LECSFormer import LECSFormer
from config import get_config
from collections import OrderedDict
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                     default='/home/ai3/student/zhaonan/crack_dataset/CrackLS315_ori_test_40',
                    # default='/home/ai3/student/zhaonan/crack_dataset/ct260_crop_test_200',
                    help='data dir')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='output', help='output dir')
parser.add_argument('--checkpoints',type=str, default='output/crackls315/epoch_69.pth', help='weights')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=list, default=[512,512], help='input patch size of network input')
parser.add_argument('--seed', type=int, default=44, help='random seed')
parser.add_argument('--cfg', type=str, default='configs/config.yaml', metavar="FILE", help='path to config file', )
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ' 'full: cache all data,' 'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
args = parser.parse_args()
config = get_config(args)

def inference(args, model, test_save_path=None):
    test_data = Crack_Datasets(data_root=args.root_path,
                               img_list=os.path.join(args.root_path, 'test.txt'),
                               img_size=args.img_size,
                               mode='test'
                               )
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['name'][0]
        with torch.no_grad():
            outputs, mid_fea = model(image)
            out = torch.sigmoid(outputs).squeeze(0)
            out = out.detach().cpu().numpy()
            cv2.imwrite(os.path.join(test_save_path, case_name + '.png'), out.squeeze(0)*255)
    return "Testing Finished!"

if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = LECSFormer(img_size=args.img_size,
                    patch_size=config.MODEL.LECSFormer.PATCH_SIZE,
                    in_channels=config.MODEL.LECSFormer.IN_CHANS,
                    num_classes=args.num_classes,
                    embed_dim=config.MODEL.LECSFormer.EMBED_DIM,
                    depths=config.MODEL.LECSFormer.DEPTHS,
                    num_heads=config.MODEL.LECSFormer.NUM_HEADS,
                    window_size=config.MODEL.LECSFormer.WINDOW_SIZE,
                    mlp_ratio=config.MODEL.LECSFormer.MLP_RATIO,
                    qkv_bias=config.MODEL.LECSFormer.QKV_BIAS,
                    qk_scale=config.MODEL.LECSFormer.QK_SCALE,
                    drop_rate=config.MODEL.DROP_RATE,
                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    patch_norm=config.MODEL.LECSFormer.PATCH_NORM,
                    use_checkpoint=config.TRAIN.USE_CHECKPOINT
                    ).cuda()

    model = nn.DataParallel(model)
    check_points = torch.load(args.checkpoints)

    # new_ckpt = OrderedDict()
    # for k,v in check_points.items():
    #     name = k[7:]
    #     new_ckpt[name] = v
    msg = model.load_state_dict(check_points,strict=True)
    checkpoints_name = args.checkpoints.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+ checkpoints_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(checkpoints_name)

    test_save_path = os.path.join(args.output_dir, "test")
    os.makedirs(test_save_path, exist_ok=True)
    inference(args, model, test_save_path)


