import os
import torch
from runpy import run_path

def load_model(task='lowlight_enhancement'):
    parameters = {
        'inp_channels': 3,
        'out_channels': 3,
        'n_feat': 80,
        'chan_factor': 1.5,
        'n_RRG': 4,
        'n_MRB': 2,
        'height': 3,
        'width': 2,
        'bias': False,
        'scale': 1,
        'task': task
    }

    # if task == 'lowlight_enhancement':
    weights = os.path.join('MIRNetv2', 'Enhancement', 'pretrained_models', 'enhancement_lol.pth')

    load_arch = run_path(os.path.join('MIRNetv2', 'basicsr', 'models', 'archs', 'mirnet_v2_arch.py'))
    model = load_arch['MIRNet_v2'](**parameters)
    model.cpu()

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    model.eval()

    return model

def enhance_image(model, image_path, output_path):
    import cv2
    import torch.nn.functional as F
    from skimage import img_as_ubyte
    import numpy as np

    img_multiple_of = 4

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).cpu()

    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - h if h % img_multiple_of != 0 else 0
    padw = W - w if w % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)
        restored = restored[:, :, :h, :w]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

    cv2.imwrite(output_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))