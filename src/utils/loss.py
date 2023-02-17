from typing import Iterable, Iterator
import torch
import torch.nn as nn
from torchvision.models import vgg16


class LatentRegularizerLoss(nn.Module):
    def __init__(self, device, reg_lambda=1e-5, step_decay=10, decay_rate=1.):
        super(LatentRegularizerLoss, self).__init__()
        self.device = device
        self.reg_lambda = reg_lambda
        self.image_loss = nn.MSELoss()
        self.step_decay = step_decay
        self.decay_rate = decay_rate
        self._step = 0

    def forward(self, latent_z, pred_images, true_images):
        # latent_z: [batch, latent_dim]
        # pred_images: [batch, n_stack, in_channels, height, width]
        # true_images: [batch, n_stack, in_channels, height, width]
        loss_img = self.image_loss(pred_images, true_images)
        # print("Shape must be [batch, n_stack, in_channels, height, width]")
        # print(pred_images.shape, true_images.shape)
        loss_reg = torch.linalg.norm(latent_z, ord=2, dim=-1).mean(dim=-1).mean(dim=-1)
        # print("loss_img: ", loss_img)
        # print("loss_reg: ", loss_reg)
        return loss_img + self.reg_lambda * loss_reg
    

    def step(self):
        self._step +=1
        if self._step % self.step_decay == 0:
            self.reg_lambda *= self.decay_rate
            

    def forward_print(self, latent_z, pred_images, true_images):
        # latent_z: [batch, latent_dim]
        # pred_images: [batch, n_stack, in_channels, height, width]
        # true_images: [batch, n_stack, in_channels, height, width]
        loss_img = self.image_loss(pred_images, true_images)
        loss_reg = torch.linalg.norm(latent_z, ord=2, dim=-1).mean(dim=-1).mean(dim=-1)
        print("-"*30, "Loss prints", "-"*30)
        print("loss_img: ", loss_img)
        print("loss_reg: ", self.reg_lambda * loss_reg)
        print("reg_lambda: ",self.reg_lambda)
        print("-"*73)
        return None

class CustomMSE(nn.Module):
    def __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, pred_images, true_images):
        # pred_images: [batch, n_stack, in_channels, height, width]
        # true_images: [batch, n_stack, in_channels, height, width]
        loss_img = torch.mean((pred_images - true_images)**2, dim=(1,2,3,4))
        return loss_img.mean(dim=-1)


class ImageFocusLatentRegularizerLoss(nn.Module):
    def __init__(self, device, reg_lambda=1e-5, encoding_lambda=0.01, step_decay=10, decay_rate=1.):
        super(ImageFocusLatentRegularizerLoss, self).__init__()
        self.device = device
        self.reg_lambda = reg_lambda
        self.encoding_lambda = encoding_lambda
        self.image_loss = CustomMSE()
        self.step_decay = step_decay
        self.decay_rate = decay_rate
        self._step = 0

    def forward(self, latent_z, preds, trues):
        # latent_z: [batch, latent_dim]
        # pred_images: [batch, n_stack, in_channels, height, width]
        # true_images: [batch, n_stack, in_channels, height, width]
        # print(preds.shape, trues.shape)

        c = preds.shape[2]

        assert c == trues.shape[2]

        pred_image = preds[:, :, 0, :, :].unsqueeze(2)
        true_image = trues[:, :, 0, :, :].unsqueeze(2)

        loss_img = self.image_loss(pred_image, true_image)
        # print("Shape must be [batch, n_stack, in_channels, height, width]")
        # print(pred_images.shape, true_images.shape)
        loss_reg = torch.linalg.norm(latent_z, ord=2, dim=-1).mean(dim=-1).mean(dim=-1)
        # print("loss_img: ", loss_img)
        # print("loss_reg: ", loss_reg)

        if c > 1:
            pred_encoding = preds[:, :, 1:, :, :]
            true_encoding = trues[:, :, 1:, :, :]
            loss_encoding = self.image_loss(pred_encoding, true_encoding)
            return loss_img + self.encoding_lambda * loss_encoding + self.reg_lambda * loss_reg
    
        return loss_img + self.reg_lambda * loss_reg

    def step(self):
        self._step +=1
        if self._step % self.step_decay == 0:
            self.reg_lambda *= self.decay_rate
            

    def forward_print(self, latent_z, preds, trues):
        # latent_z: [batch, latent_dim]
        # pred_images: [batch, n_stack, in_channels, height, width]
        # true_images: [batch, n_stack, in_channels, height, width]
        pred_image = preds[:, :, 0, :, :].unsqueeze(2)
        true_image = trues[:, :, 0, :, :].unsqueeze(2)
        pred_encoding = preds[:, :, 1:, :, :]
        true_encoding = trues[:, :, 1:, :, :]

        loss_img = self.image_loss(pred_image, true_image)
        loss_encoding = self.image_loss(pred_encoding, true_encoding)
        # print("Shape must be [batch, n_stack, in_channels, height, width]")
        # print(pred_images.shape, true_images.shape)
        loss_reg = torch.linalg.norm(latent_z, ord=2, dim=-1).mean(dim=-1).mean(dim=-1)
        print("-"*30, "Loss prints", "-"*30)
        print("loss_img: ", loss_img)
        print("loss_encoding: ", self.encoding_lambda * loss_encoding)
        print("loss_reg: ", self.reg_lambda * loss_reg)
        print("reg_lambda: ",self.reg_lambda)
        print("-"*73)
        return None


class LossPerceptual(torch.nn.Module):
    def __init__(self, device=None, vgg_model=None, loss=nn.L1Loss()):
        super(LossPerceptual, self).__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if vgg_model is None:
            self.vgg_model = vgg16(pretrained=True).eval().to(device)
        self.vgg_layers = self.vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        self.loss = loss

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def forward(self, latent_z, pred, gt, normalized=False):
        b, n, c, h, w = pred.shape
        if not normalized:
            gt = (gt - self.mean) / self.std
            pred = (pred - self.mean) / self.std

            # print("gt: ", gt.shape)
            # print("pred: ", pred.shape)
            # make them 3 channels
            gt = gt[:,:,0].unsqueeze(2)
            gt = gt.reshape(-1, c, h, w)
            pred = pred[:,:,0].unsqueeze(2)
            pred = pred.reshape(-1, c, h, w)
            # print("gt: ", gt.shape)
            # print("pred: ", pred.shape)

            gt = gt.repeat(1, 3, 1, 1)
            pred = pred.repeat(1, 3, 1, 1)
            # print("gt: ", gt.shape)
            # print("pred: ", pred.shape)

        loss = 0
        for name, module in self.vgg_layers._modules.items():
            pred = module(pred)
            gt = module(gt)
            if name in self.layer_name_mapping:
                loss += self.loss(pred, gt)

        return loss


class LossWrapper(nn.Module):
    def __init__(self, losses, weights):
        super(LossWrapper, self).__init__()

        assert isinstance(losses, Iterable)
        assert len(losses) == len(weights)
        
        self.losses = losses
        self.weigths = weights

    def forward(self, latent_z, preds, trues):
        loss = []
        for i, loss_func in enumerate(self.losses):
            loss.append(self.weigths[i] * loss_func(latent_z, preds, trues))
            
        print(loss)
        return sum(loss)
        # return loss
