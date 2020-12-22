#!/usr/bin/env python
# coding: utf-8

# Script to generate grayscale image data. With the input arguments one can specify
# how many gabor patches should be on the image, the image height as well as the
# number of noise patches on the image. The metadata of the images are saved in a
# file named description.csv, containing the image name and the according orientations.
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
                        help="The number of gabor patches on one image")
    parser.add_argument('--n_noise_patches', type=int, default=15,
                        help="The number of gaussian noise patches in the image")
    parser.add_argument('--image_height', type=int, default=250,
                        help='Height of the image. The width will be n_gabor_patches time the height')
    parser.add_argument('--output_path', type=str, default='images/',
                        help="Location for saving the images")
    parser.add_argument('--n_images', type=int, default=100,
                        help="The number of images generated")
    return parser.parse_args()


# this method is a slightly adjusted copy from
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
"""


def gabor_patch(size, lambda_, theta, sigma, phase, trim=.005):
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
    img_data = (grating * gauss + 1) / 2 * 255
    return Image.fromarray(img_data)


def generate_image(image_height, n_gabor_patches, patch_size):
    background_color = "#7f7f7f"
    lambda_ = 20
    sigma = 30

    total_img = Image.new(
        "L",
        (image_height *
         n_gabor_patches,
         image_height),
        background_color)
    orientations = []
    for i in range(0, n_gabor_patches):
        orientation = np.random.uniform(0, 180)
        orientations.append(orientation)
        phase = np.random.uniform(0, 360)
        patch = gabor_patch(
            int(patch_size),
            lambda_,
            orientation,
            sigma,
            phase)
        total_img.paste(patch,
                        (int((image_height - patch_size) / 2 + i * image_height),
                         int((image_height - patch_size) / 2)))
    return total_img, orientations


def add_noise_patch(img, diameter=50):
    diam = round(diameter)
    img_width, img_height = img.size
    center_x = round(np.random.uniform(0, img_width))
    center_y = round(np.random.uniform(0, img_height))
    radius = round(diameter / 2)
    noise_square = np.clip(
        (np.random.normal(
            loc=0.5, scale=0.1, size=(
                diam, diam))), 0, 1) * 255
    start_x = center_x - radius
    start_y = center_y - radius

    for x in range(0, diam):
        for y in range(0, diam):
            coord_x = start_x + x
            coord_y = start_y + y
            if coord_x > 0 and coord_x < img_width and coord_y > 0 and coord_y < img_height and (
                    (x - radius) ** 2 + (y - radius) ** 2) < radius ** 2:
                img.putpixel((coord_x, coord_y), int(noise_square[x, y]))
    return img


def add_noise_patches(img, number=5, max_diameter=70, min_diameter_scale=0.8):
    for _ in range(0, number):
        img = add_noise_patch(
            img,
            diameter=max_diameter *
            np.random.uniform(
                min_diameter_scale,
                1))
    return img


def generate_noisy_image(n_gabor_patches, n_noise_patches, image_height):
    patch_size = image_height * 0.8
    img, orientations = generate_image(
        image_height, n_gabor_patches, patch_size)
    noisy_img = add_noise_patches(img, n_noise_patches, patch_size / 3)
    return noisy_img, orientations


if __name__ == '__main__':
    FLAGS = get_args()
    number_images = FLAGS.n_images
    n_gabor_patches = FLAGS.n_gabor_patches
    n_noise_patches = FLAGS.n_noise_patches
    output_path = FLAGS.output_path
    image_height = FLAGS.image_height

    Path(output_path).mkdir(parents=True, exist_ok=True)

    columns = ["image_name"]
    for i in range(0, n_gabor_patches):
        columns.append("orientation_%d" % i)
    df = pd.DataFrame(columns=columns)

    for i in range(0, number_images):
        img, orientations = generate_noisy_image(
            n_gabor_patches, n_noise_patches, image_height)
        img_name = ("gabor%d_%06d.png" % (n_gabor_patches, i))
        row = [img_name]
        row.extend(orientations)
        df = df.append(pd.Series(row, index=columns), ignore_index=True)
        img.save(output_path + img_name)
    df.to_csv(output_path + "description.csv")
