import numpy as np
from tqdm import tqdm
import torch
from imageio import imread, imwrite
from path import Path
import os

from config import get_opts, get_training_size

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2
from SC_DepthV3 import SC_DepthV3

import datasets.custom_transforms as custom_transforms

from visualization import *


@torch.no_grad()
def main():
    hparams = get_opts()

    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)
    elif hparams.model_version == 'v3':
        system = SC_DepthV3(hparams)

    hparams.ckpt_path = './ckpts/nyu_scv3/epoch=93-val_loss=0.1384.ckpt'
    # hparams.ckpt_path = './ckpts/epoch=75-val_loss=0.1438.ckpt'
    system = SC_DepthV3.load_from_checkpoint(hparams.ckpt_path, strict=False)

    model = system.depth_net
    model.cuda()
    model.eval()

    # training size
    training_size = get_training_size(hparams.dataset_name)

    # normalization
    inference_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )

    input_dir = Path(hparams.input_dir)
    output_dir = Path(hparams.output_dir) / \
        'model_{}'.format(hparams.model_version)
    output_dir.makedirs_p()

    if hparams.save_vis:
        (output_dir/'vis').makedirs_p()

    if hparams.save_depth:
        (output_dir/'depth').makedirs_p()

    image_files = sum([(input_dir).files('*.{}'.format(ext))
                      for ext in ['jpg', 'png']], [])
    image_files = sorted(image_files)

    print('{} images for inference'.format(len(image_files)))

    for i, img_file in enumerate(tqdm(image_files)):

        filename = os.path.splitext(os.path.basename(img_file))[0]

        img = imread(img_file).astype(np.float32)
        tensor_img = inference_transform([img])[0][0].unsqueeze(0).cuda()
        dummy_input = tensor_img
        pred_depth = model(tensor_img)

        if hparams.save_vis:
            vis = visualize_depth(pred_depth[0, 0]).permute(
                1, 2, 0).numpy() * 255
            imwrite(output_dir/'vis/{}.jpg'.format(filename),
                    vis.astype(np.uint8))

        if hparams.save_depth:
            depth = pred_depth[0, 0].cpu().numpy()
            np.save(output_dir/'depth/{}.npy'.format(filename), depth)

    device = torch.device("cuda")
    #dummy_rand = torch.randn(1, 3, 640, 480).to(device)  # Adjust input size as needed

    model_name = "sc_depth_v3_nyu.onnx"
    torch.onnx.export(model,  # model being run
                    tensor_img,  # model input (or a tuple for multiple inputs)
                    model_name,  # where to save the model (can be a file or file-like object)
                    input_names=['input'],
                    output_names=['output'],
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=16)
    
    print("\n-> Done!")

if __name__ == '__main__':
    main()
