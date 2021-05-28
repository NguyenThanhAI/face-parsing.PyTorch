import argparse
import os
from PIL import Image

import numpy as np
import cv2

import torch
from torchvision import transforms

import matplotlib.pyplot as plt

from model import BiSeNet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_pth", type=str, default="res/cp/79999_iter.pth")
    parser.add_argument("--image_path", type=str, default=None)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    save_pth = args.save_pth
    image_path = args.image_path

    n_classes = 19
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    net = BiSeNet(n_classes=n_classes)

    net.load_state_dict(torch.load(save_pth, map_location=torch.device("cpu")))

    net.to(device)
    net.eval()

    image = Image.open(image_path)
    #image = image.resize((1024, 1024), Image.BILINEAR)
    with torch.no_grad():
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = net(img)[0]

    if torch.cuda.is_available():
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    else:
        parsing = out.squeeze(0).numpy().argmax(0)

    image = np.array(image)

    vis_mask = np.array([list(map(lambda x: part_colors[x], row)) for row in parsing]).astype(np.uint8)

    vis_img = cv2.addWeighted(image[:, :, ::-1], 0.4, vis_mask, 0.6, 0)

    cv2.imshow("Anh", vis_img)
    cv2.waitKey(0)

    for i in range(n_classes):
        vis_mask = np.array([list(map(lambda x: part_colors[x] if x == i else [0, 0, 0], row)) for row in parsing]).astype(np.uint8)

        vis_img = cv2.addWeighted(image[:, :, ::-1], 0.4, vis_mask, 0.6, 0)

        cv2.imshow(str(i), vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
