#!/usr/bin/env python
# coding: utf-8

# Script to generate colored image data. With the input arguments one can specify
# how many gabor patches should be on the image, he image resolution as well as the
# number of noise patches on the image. Furthermore one can add noise to the color
# of the patch by defining the standard deviation of color_noise. The metadata of
# the images are saved in a file named description.csv, containing the image name
# together with the according orientations and color values.
# Please note that the generation of the noise patches is quite slow and if somebody
# comes up with a more efficient method, a pull request is highly appreciated.

import argparse

from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gabor_patches', type=int, default=3,
                        help="The number of gabor patches on the image")
    parser.add_argument('--n_noise_patches', type=int, default=24,
                        help="The number of gaussian noise patches in the image")
    parser.add_argument('--color_noise', type=float, default=0,
                        help="Standard deviation of the gaussian to add noise to the gabor patches (in degrees)")
    parser.add_argument('--image_height', type=int, default=250,
                        help='Height of the image. The width will be n_gabor_patches time the height')
    parser.add_argument('--output_path', type=str, default='images/',
                        help="Location for saving the images")
    parser.add_argument('--n_images', type=int, default=100,
                        help="The number of images generated")
    return parser.parse_args()


# this method is based on this stackoverflow post
# https://stackoverflow.com/questions/19039674/how-can-i-expand-this-gabor-patch-to-the-size-of-the-bounding-box
"""
    lambda_ : int
        Spatial frequency (px per cycle)

    theta : int or float
        Grating orientation in degrees (0-180)

    sigma : int or float
        gaussian standard deviation (in pixels)

    phase : float
        phase offset of the grating, between 0 and 180

    trim : float
        used to cut the gaussian at some point
        preventing it from continuing infinitely

    color_trim : float
        defines the color border, compare to trim

    color_std_dev : int or float
        if greater than 0, the given color is interpreted
        as the mean of a gaussian function and color_std_dev
        is the standard deviation, used to sample for every pixel
"""


def gabor_patch(size, color, lambda_, theta, sigma, phase, trim=.005,
                color_trim=.005, color_std_dev=20):
    X0 = (np.linspace(1, size, size) / size) - .5
    freq = size / float(lambda_)
    phaseRad = (phase / 180.) * np.pi
    Xm, Ym = np.meshgrid(X0, X0)
    thetaRad = (theta / 180.) * np.pi
    Xt = Xm * np.cos(thetaRad)
    Yt = Ym * np.sin(thetaRad)
    grating = np.sin(((Xt + Yt) * freq * 2 * np.pi) + phaseRad)
    gauss = np.exp(-((Xm ** 2) + (Ym ** 2)) / (2 * (sigma / float(size)) ** 2))
    gauss[gauss < trim] = 0
    mask = np.where(gauss < color_trim, True, False)  # where should it be cut?
    color_pixel = color / 360 * 255
    color_std_dev_pixel = color_std_dev / 360 * 255
    v = (1 - (grating * gauss + 1) / 2) * 255
    h = np.ones(v.shape) * color_pixel
    if color_std_dev_pixel > 1e-3:
        h = np.random.normal(h, color_std_dev_pixel)
    s = np.clip(np.ones(v.shape) * gauss * 2, 0, 1) * 255
    h[mask] = 0
    s[mask] = 0
    v[gauss == 0] = 127
    img_data = np.transpose(np.array([h, s, v]))
    img_data = np.uint8(img_data)
    return Image.fromarray(img_data, "HSV").convert("RGB")


def generate_image(orientations, colors, image_height,
                    patch_size, color_std_dev):
    background_color = "#7f7f7f"
    n_gabor_patches = len(orientations)
    img = Image.new("RGB", (image_height * n_gabor_patches, image_height), background_color)
    lambda_ = 20
    sigma = 30
    for i, (o, c) in enumerate(zip(orientations, colors)):
        phase = np.random.uniform(0, 360)
        patch = gabor_patch(int(patch_size), c, lambda_, o, sigma, phase, color_std_dev=color_std_dev)
        img.paste(patch, (int((image_height - patch_size) / 2 + i * image_height), int((image_height - patch_size) / 2)))   
    return img


def generate_random_image(image_height, patches, patch_size, color_std_dev):
    orientations = []
    colors = []
    for _ in range(0, patches):
        orientation = np.random.uniform(0, 180)
        color = np.random.uniform(0, 360)
        orientations.append(orientation)
        colors.append(color)
    img = generate_image(orientations, colors, image_height, patch_size, color_std_dev) 
    return img, orientations, colors


def add_noise_patch(img, diameter=50, center=(None, None)):
    diam = round(diameter)
    radius = round(diameter / 2)
    img_width, img_height = img.size

    center_x, center_y = center
    if center_x is None:
        center_x = round(np.random.uniform(0, img_width))
    if center_y is None:
        center_y = round(np.random.uniform(0, img_height))

    h = np.random.uniform(0, 255, (diam, diam))
    s = np.clip((np.random.normal(loc=0.5, scale=0.1, size=(diam, diam))), 0, 1) * 255
    v = np.clip((np.random.normal(loc=0.5, scale=0.1, size=(diam, diam))), 0, 1) * 255

    start_x = center_x - radius
    start_y = center_y - radius

    for x in range(0, diam):
        for y in range(0, diam):
            coord_x = start_x + x
            coord_y = start_y + y
            if coord_x > 0 and coord_x < img_width and coord_y > 0 and coord_y < img_height and (
                    (x - radius) ** 2 + (y - radius) ** 2) < radius ** 2:
                img.putpixel((coord_x, coord_y), (int(h[x, y]), int(s[x, y]), int(v[x, y])))
    return img


def add_noise_patches(img, number=5, max_diameter=70, min_diameter_scale=0.8):
    for _ in range(0, number):
        img = add_noise_patch(img, diameter=max_diameter * np.random.uniform(min_diameter_scale, 1))
    return img


def generate_noisy_image(n_gabor_patches, n_noise_patches, image_height, color_std_dev):
    patch_size = image_height * 0.8
    img, orientations, colors = generate_random_image(image_height, n_gabor_patches, patch_size, color_std_dev)
    noisy_img = add_noise_patches(img.convert("HSV"), n_noise_patches, patch_size / 3).convert("RGB")
    return noisy_img, orientations, colors


if __name__ == '__main__':
    FLAGS = get_args()
    number_images = FLAGS.n_images
    n_gabor_patches = FLAGS.n_gabor_patches
    n_noise_patches = FLAGS.n_noise_patches
    color_std_dev = FLAGS.color_noise
    output_path = FLAGS.output_path
    image_height = FLAGS.image_height

    Path(output_path).mkdir(parents=True, exist_ok=True)

    columns = ["image_name"]
    for i in range(0, n_gabor_patches):
        columns.append("orientation_%d" % i)
    for i in range(0, n_gabor_patches):
        columns.append("color_%d" % i)
    df = pd.DataFrame(columns=columns)

    for i in range(0, number_images):
        img, orientations, colors = generate_noisy_image(n_gabor_patches, n_noise_patches, image_height, color_std_dev)
        img_name = ("gabor%d_%06d.png" % (n_gabor_patches, i))
        row = [img_name]  # just a list
        row.extend(orientations)
        row.extend(colors)
        df = df.append(pd.Series(row, index=columns), ignore_index=True)
        img.save(output_path + img_name)
    df.to_csv(output_path + "description.csv")
