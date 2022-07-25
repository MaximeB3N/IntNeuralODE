import torch
import torch.nn as nn


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

class ImageFocusLatentRegularizerLoss(nn.Module):
    def __init__(self, device, reg_lambda=1e-5, encoding_lambda=0.01, step_decay=10, decay_rate=1.):
        super(ImageFocusLatentRegularizerLoss, self).__init__()
        self.device = device
        self.reg_lambda = reg_lambda
        self.encoding_lambda = encoding_lambda
        self.image_loss = nn.MSELoss()
        self.step_decay = step_decay
        self.decay_rate = decay_rate
        self._step = 0

    def forward(self, latent_z, preds, trues):
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
        # print("loss_img: ", loss_img)
        # print("loss_reg: ", loss_reg)
        return loss_img + self.encoding_lambda * loss_encoding + self.reg_lambda * loss_reg
    

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
