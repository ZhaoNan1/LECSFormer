import argparse
from distutils.command.config import config
import os
import random
from symbol import parameters
from webbrowser import get
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import get_config
from networks.LECSFormer import LECSFormer
from trainer import trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, 
                        default='/home/ai3/student/zhaonan/crack_dataset/crackls315', 
                        help='data dir')
    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    parser.add_argument('--output_dirs', type=str, default='output/crackls315', help='output dir')
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.001, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=list, default=[512,512], help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=44, help='random seed')
    parser.add_argument('--cfg', type=str, default='configs/config.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument('--resume',help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help=" 'no: no cache,' 'full: cache all data,' 'part: sharding the dataset into nonoverlapping pieces and only cache one piece' ")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)

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

    if not os.path.exists(args.output_dirs):
        os.makedirs(args.output_dirs)

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


    trainer(args, model, config)