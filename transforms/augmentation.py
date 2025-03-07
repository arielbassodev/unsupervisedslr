import random
import numpy as np
import math
from math import sin,cos
import random
import  torch
from torch import nn
import torch.nn.functional as F
import os
import sys
import torchvision
import random
import torch
import random
from sign_language_tools.pose.transform import Rotation2D, translation, flip, smooth, noise, interpolate, padding, scale
from lsfb_transfo.models.utils import gaus_noise


class Translation:
    def __init__(self, dx: float, dy: float):
        self.dx = dx
        self.dy = dy

    def __call__(self, landmarks: torch.Tensor):
        translation_vector = torch.tensor([self.dx, self.dy], dtype=landmarks.dtype, device=landmarks.device)
        return landmarks + translation_vector

class RandomTranslation(Translation):
    def __init__(self, dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2)):
        dx = random.uniform(*dx_range)
        dy = random.uniform(*dy_range)
        super().__init__(dx, dy)

class Translate:
    def __init__(self, dx, dy):
        self.translation = Translation(dx, dy)

    def __call__(self, landmark):
        shape = landmark.shape
        translated = torch.stack([self.translation(l) for l in landmark])
        return translated.to(landmark.device)

class Rotation2D:
    def __init__(self, angle):
        theta = np.radians(angle)
        self.rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ], dtype=torch.float32)

    def __call__(self, points):
        rotation_matrix = self.rotation_matrix.to(points.dtype)
        rotation_matrix =  rotation_matrix.to('cuda')
        return torch.einsum("ij,hwj->hwi", rotation_matrix, points)

class Rotate:
    def __init__(self, angle):
        self.angle = angle
        self.rotation = Rotation2D(angle)

    def __call__(self, landmark):
        shape = landmark.shape
        landmark =  landmark.to('cuda')
        rotated = torch.stack([self.rotation(l) for l in landmark])
        return rotated.to(landmark.device)

class GaussianNoise:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, landmarks: torch.Tensor):
        shape = landmarks.shape
        noisy = torch.stack([self.add_noise(l) for l in landmarks])
        return noisy.to(landmarks.device)

    def add_noise(self, landmark: torch.Tensor):

        noise = torch.randn_like(landmark) * self.scale
        return landmark + noise

class HorizontalFlip:
    def __call__(self, landmarks: torch.Tensor):
        shape = landmarks.shape
        flipped = torch.stack([self.flip(l) for l in landmarks])
        return flipped.to(landmarks.device)

    def flip(self, landmark: torch.Tensor):
        landmark[..., 0] = 1 - landmark[..., 0]
        return landmark

class VerticalFlip:
    def __call__(self, landmarks: torch.Tensor):
        shape = landmarks.shape
        flipped = torch.stack([self.flip(l) for l in landmarks])
        return flipped.to(landmarks.device)

    def flip(self, landmark: torch.Tensor):

        landmark[..., 1] = 1 - landmark[..., 1]
        return landmark

def apply_random_augmentation():
    vf = VerticalFlip()
    hf = HorizontalFlip()
    noise = GaussianNoise(0.4)
    rotation = Rotate(30)
    translation = Translate(0.05, 0.05)
    list_aug =  [Rotate(30), Rotate(15), Rotate(45), Rotate(60),Translate(0.1, 0.1), GaussianNoise(0.1), GaussianNoise(0.2), GaussianNoise(0.3), GaussianNoise(0.4), Translate(0.35, 0.2), Translate(-0.2, 0.3) ]
    return random.choice(list_aug)
    #Rotate(30), Rotate(15), Rotate(45), Rotate(60), Rotate(90), Rotate(120), Rotate(150), Rotate(180), Rotate(210), Rotate(240), Rotate(270), Rotate(300), Rotate(330), Rotate(10), Rotate(25), Rotate(35), Rotate(55), Rotate(75), Rotate(105), Rotate(135), Rotate(165), Rotate(195)
    #Translate(0.2, 0), Translate(0.3, 0), Translate(0.4, 0), Translate(0.5, 0), Translate(0.6, 0), Translate(0.7, 0), Translate(0.8, 0), Translate(0.1, 0.1), Translate(0.25, -0.1), Translate(0.35, 0.2), Translate(-0.2, 0.3), Translate(0.5, -0.5), Translate(0.6, 0.4), Translate(-0.3, -0.2), Translate(0.45, 0.1), Translate(0.75, -0.3), Translate(0.9, 0.5), Translate(-0.1, -0.1), Translate(0.2, 0.25), Translate(0.55, -0.4), Translate(0.68, 0.3), Translate(0.8, -0.6)

