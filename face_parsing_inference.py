import argparse
import os
from PIL import Image

import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from model import BiSeNet

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, device, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def initialize_model(vgg_checkpoint, decoder_checkpoint, device):
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_checkpoint))
    vgg.load_state_dict(torch.load(vgg_checkpoint))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    return vgg, decoder


def run_forward(content, style, content_tf, style_tf, vgg, decoder, device,
                preserve_color, alpha, interpolation_weights):
    assert content.shape[2] == 3 and style.shape[2] == 3
    content = Image.fromarray(content)
    style = Image.fromarray(style)
    content = content_tf(content)
    style = style_tf(style)

    if preserve_color:
        style = coral(style, content)

    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)

    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style, device, alpha, interpolation_weights)

    if torch.cuda.is_available():
        output = output.cpu()

    output = make_grid(output)
    output = output.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    return output


def initialize_parsing_model(n_classes, save_pth, device):
    net = BiSeNet(n_classes=n_classes)

    net.load_state_dict(torch.load(save_pth, map_location=torch.device("cpu")))

    net.to(device)
    net.eval()

    return net


def run_parsing_forward(net, image):
    with torch.no_grad():
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = net(img)[0]

    if torch.cuda.is_available():
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    else:
        parsing = out.squeeze(0).numpy().argmax(0)

    return parsing


def run_inference(image, overlay, content_tf, style_tf, vgg, decoder, device,
                  preserve_color, alpha, interpolation_weights):
    #image = image.resize((1024, 1024), Image.BILINEAR)
    parsing = run_parsing_forward(net=net, image=image)

    image = np.array(image)
    overlay = overlay.resize((image.shape[1], image.shape[0]))
    overlay = np.array(overlay)

    #parsing = cv2.resize(parsing, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
    #image = cv2.resize(image, (overlay.shape[1], overlay.shape[0]))

    hair_mask = np.where(np.isin(parsing, [16, 17, 18]), True, False)
    skin_mask = np.where(np.isin(parsing, list(range(1, 16))), True, False)
    mask = np.where(hair_mask[:, :, np.newaxis], image, np.zeros_like(image))
    #cv2.imshow("Mask", mask)
    #cv2.waitKey(0)

    stylized_hair = run_forward(mask, overlay, content_tf, style_tf, vgg, decoder, device,
                                preserve_color, alpha, interpolation_weights)

    stylized_hair = cv2.resize(stylized_hair, (mask.shape[1], mask.shape[0]))

    stylized_hair = np.where(parsing[:, :, np.newaxis], stylized_hair, overlay)

    #cv2.imshow("Output", stylized_hair[:, :, ::-1])
    #cv2.waitKey(0)

    kernel_size = max(max(image.shape[0], image.shape[1]) // 5, 31)
    if kernel_size % 2 == 0:
        kernel_size += 1

    skin_mask = np.where(skin_mask, 0.6, 0.4)
    smoothed_skin_mask = cv2.GaussianBlur(skin_mask, (kernel_size, kernel_size), sigmaX=20., sigmaY=20.)

    result = smoothed_skin_mask[:, :, np.newaxis] * image + (1 - smoothed_skin_mask[:, :, np.newaxis]) * stylized_hair
    #cv2.imwrite("result.jpg", result[:, :, ::-1])
    #cv2.imshow("Anh", result[:, :, ::-1])
    #cv2.waitKey(0)

    return result


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_pth", type=str, default="res/cp/79999_iter.pth")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--overlay_path", type=str, default=None)

    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='models/decoder.pth')

    parser.add_argument('--content_size', type=int, default=0,
                        help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
    parser.add_argument('--style_size', type=int, default=0,
                        help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--preserve_color', action='store_true',
                        help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
    parser.add_argument(
        '--style_interpolation_weights', type=str, default=None,
        help='The weight for blending the style of multiple style images')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    save_pth = args.save_pth
    image_path = args.image_path
    overlay_path = args.overlay_path

    vgg = args.vgg
    decoder = args.decoder

    content_size = args.content_size
    style_size = args.style_size
    crop = args.crop
    preserve_color = args.preserve_color
    alpha = args.alpha

    if args.style_interpolation_weights is not None:
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
    else:
        interpolation_weights = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vgg, decoder = initialize_model(vgg_checkpoint=vgg, decoder_checkpoint=decoder, device=device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    n_classes = 19

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    net = initialize_parsing_model(n_classes=n_classes, save_pth=save_pth, device=device)

    image = Image.open(image_path)
    overlay = Image.open(overlay_path)

    result = run_inference(image=image, overlay=overlay, content_tf=content_tf, style_tf=style_tf,
                           vgg=vgg, decoder=decoder, device=device, preserve_color=preserve_color,
                           alpha=alpha, interpolation_weights=interpolation_weights)

    cv2.imwrite("result.jpg", result[:, :, ::-1])
