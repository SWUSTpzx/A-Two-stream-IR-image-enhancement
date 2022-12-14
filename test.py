import torch
import os
from utils import cal_psnr
import numpy as np
import cv2
import glob
from torchvision import transforms
from PIL import Image
from skimage.measure import compare_ssim
from D_InfENet import Double

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_images(): 
    net = Double()
    train_model = "./weights.pth"
    net.load_state_dict(torch.load(train_model, map_location=device))
    net.eval()
    print('loading model')

    all_path = './all'
    if not os.path.exists(all_path):
        os.makedirs(all_path)
    test_img_dir = "BSDS200/*.png"



    files = glob.glob(test_img_dir)

    transform_high = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), ])
    transform_low = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ColorJitter(contrast=[0.5, 0.51]),
        transforms.ToTensor(), ])

    psnr_ = 0
    # min_ = 100
    # max_ = 0
    ss, s = 0, 0
    n = 0

    for f_ in files:
        img_high = transform_high(Image.open(str(f_)))
        image_low = transform_low(Image.open(str(f_)))

        image_low_gary = torch.unsqueeze(image_low, 0)
        output = net.forward(image_low_gary)

        output = torch.squeeze(output, 0).permute(1, 2, 0).detach().numpy()
        output = output * 255

        img_high = img_high.permute(1, 2, 0).detach().numpy()
        img_high = img_high * 255

        image_low = image_low.permute(1, 2, 0).detach().numpy()
        image_low = image_low * 255

        all_img = np.concatenate([image_low, img_high, output], axis=1)


        r = cal_psnr(output, img_high, 255.)
        s = compare_ssim(output, img_high, multichannel=True)

        ss += s
        psnr_ += r
        n += 1
        # if r > max_: max_ = r
        # if r < min_: min_ = r

        print(f'num:{n}, psnr:{r}, ssim:{s}')
        cv2.imwrite(all_path + "/" + str(n) + '.png', all_img)


    print('total_num:', len(files), 'psnr:', psnr_ / len(files))
    print('ssim_avg:', ss / len(files))


if __name__ == "__main__":

    test_images()