from model import MattingBase, MattingRefine
from torch.optim import Adam

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'segmenter'))
from segmenter.segm.model import factory

sys.path.append(os.path.join(os.path.dirname(__file__), 'Uformer'))
from Uformer.utils.model_utils import get_arch

def load_model_and_optimizer(args, device):
    if args.model_backbone in ['resnet101', 'resnet50', 'mobilenetv2']:
        model = MattingBase(args.model_backbone).to(device)
        optimizer = Adam([
            {'params': model.backbone.parameters(), 'lr': 1e-4},
            {'params': model.aspp.parameters(), 'lr': 5e-4},
            {'params': model.decoder.parameters(), 'lr': 5e-4}
        ])
    elif args.arch is not None:
        model = get_arch(args)
        optimizer = Adam(model.parameters(), lr=1e-4)
    else:
        model_cfg = factory.create_model_cfg(args)
        model = factory.create_segmenter(model_cfg).to(device)
        optimizer = Adam(model.parameters(), lr=1e-4)

    return model, optimizer