import torch.nn as nn
from tqdm import trange

from .viz import display_ode_trajectory
from .metrics import PSNR, SSIM


def train_convnode(model, optimizers, scheduler, epochs, getter, display=100, loss_fn=None, display_results_fn=display_ode_trajectory, out_display=-1):
    """
    Train function for a ConvNode like model.

    Parameters
    ----------
    model : nn.Module, the model to train
    optimizer : torch.optim, the optimizer to use
    epochs : int, the number of epochs to train for
    getter : utils.data.DataGetter, the data getter to use
    display : int, the number of epochs between each display of the results
    loss_fn : nn.Module, the loss function to use, default is MSELoss
    display_results_fn : function, the function to use to display the results, default is display_ode_trajectory
    out_display : int, the number of points to display, default is -1 (all points)
    """
    device = model.device

    if out_display == -1:
        out_display = model.out_dim

    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    iterator = trange(1, epochs+1)
    # just for the plot part
    running_loss = 0.
    for i in iterator:
        # get a random time sample
        model.train()
        loss = 0.
        batch_init_images, batch_times, batch_true_images = getter.get_batch()
        batch_init_images = batch_init_images.to(device)
        batch_times = batch_times.to(device)
        batch_true_images = batch_true_images.to(device)
        # compute the output of the model
        out_images, latent = model(batch_init_images, batch_times, getter.dt)
        # compute the loss
        # print(out.shape, out.view(-1, batch_init_positions.shape[-1]).shape)
        # print(batch_true_positions.shape, batch_true_positions.view(-1, batch_init_positions.shape[-1]).shape)
        # print(out_images.shape, batch_true_images.shape)
        loss += loss_fn(latent, out_images[:,:], batch_true_images[:,:])
        # .view(-1,batch_init_positions.shape[-1])
        
        
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()
        # update the progress bar
        iterator.set_postfix_str(f'Loss: {loss.item():.8f}')
        running_loss += loss.item()

        scheduler.step()
        loss_fn.step()

        if i % display == 0:
           display_results_fn(i, model, out_display, getter, getter.total_length, getter.dt)
           iterator.set_description_str(f'Display loss: {running_loss/display:.8f}')
           running_loss = 0.
           loss_fn.forward_print(latent, out_images[:,:], batch_true_images[:,:])
        
    return None


def train_convnode_with_latent_supervision(model, optimizer, scheduler, epochs, getter, display=100, loss_fn=None, display_results_fn=display_ode_trajectory, out_display=-1):
    """
    Train function for a ConvNode like model with latent regularization.

    Parameters
    ----------
    model : nn.Module, the model to train
    optimizer : torch.optim, the optimizer to use
    epochs : int, the number of epochs to train for
    getter : utils.data.DataGetter, the data getter to use
    display : int, the number of epochs between each display of the results
    loss_fn : nn.Module, the loss function to use, default is MSELoss
    display_results_fn : function, the function to use to display the results, default is display_ode_trajectory
    out_display : int, the number of points to display, default is -1 (all points)
    """
    
    device = model.device

    if out_display == -1:
        out_display = model.out_dim

    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    iterator = trange(1, epochs+1)
    # just for the plot part
    running_loss = 0.
    for i in iterator:
        # get a random time sample
        model.train()
        loss = 0.
        batch_init_images, batch_times, batch_true_images = getter.get_batch()
        batch_init_images = batch_init_images.to(device)
        batch_times = batch_times.to(device)
        batch_true_images = batch_true_images.to(device)
        # compute the output of the model
        pred_images, pred_latent = model(batch_init_images, batch_times, getter.dt)
        true_latent = model.encode(batch_true_images)
        # compute the loss
        # print(out.shape, out.view(-1, batch_init_positions.shape[-1]).shape)
        # print(batch_true_positions.shape, batch_true_positions.view(-1, batch_init_positions.shape[-1]).shape)
        # print(out_images.shape, batch_true_images.shape)
        loss += loss_fn(pred_latent, true_latent, pred_images[:], batch_true_images[:])
        # .view(-1,batch_init_positions.shape[-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the progress bar
        iterator.set_postfix_str(f'Loss: {loss.item():.8f}')
        running_loss += loss.item()

        scheduler.step()
        loss_fn.step()

        if i % display == 0:
           display_results_fn(i, model, out_display, getter, getter.total_length, getter.dt)
           iterator.set_description_str(f'Display loss: {running_loss/display:.8f}')
           running_loss = 0.
           loss_fn.forward_print(pred_latent, true_latent, pred_images[:], batch_true_images[:])
        
    return None




# def train(model, optimizer, scheduler, epochs, batch_size, getter, display=100, loss_fn=None, display_results_fn=display_ode_trajectory, out_display=-1):
    
#     if out_display == -1:
#         out_display = model.out_dim

#     if loss_fn is None:
#         loss_fn = nn.MSELoss()
    
#     iterator = trange(1, epochs+1)
#     # just for the plot part
#     running_loss = 0.
#     for i in iterator:
#         # get a random time sample
#         model.train()
#         loss = 0.
#         for _ in range(batch_size):
#             batch_init_positions, batch_times, batch_true_positions = getter.get_batch()
#             # compute the output of the model
#             out = model(batch_init_positions, batch_times)
#             # compute the loss
#             # print(out.shape, out.view(-1, batch_init_positions.shape[-1]).shape)
#             # print(batch_true_positions.shape, batch_true_positions.view(-1, batch_init_positions.shape[-1]).shape)
#             loss += loss_fn(out[:], batch_true_positions[:])
#             # .view(-1,batch_init_positions.shape[-1])
#         loss /= batch_size
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # update the progress bar
#         iterator.set_postfix_str(f'Loss: {loss.item():.8f}')
#         running_loss += loss.item()

#         if i % display == 0:
#            display_results_fn(i, model, out_display, getter, getter.total_length, getter.dt)
#            iterator.set_description_str(f'Display loss: {running_loss/display:.8f}')
#            running_loss = 0.


#         scheduler.step()
        
#     return None

# def train_convnode(model, optimizer, scheduler, epochs, batch_size, getter, display=100, loss_fn=None, display_results_fn=display_ode_trajectory, out_display=-1):
    
#     if out_display == -1:
#         out_display = model.out_dim

#     if loss_fn is None:
#         loss_fn = nn.MSELoss()
    
#     iterator = trange(1, epochs+1)
#     # just for the plot part
#     running_loss = 0.
#     for i in iterator:
#         # get a random time sample
#         model.train()
#         loss = 0.
#         for _ in range(batch_size):
#             batch_init_images, batch_times, batch_true_images = getter.get_batch()
#             # compute the output of the model
#             out_images, _ = model(batch_init_images, batch_times, getter.dt)
#             # compute the loss
#             # print(out.shape, out.view(-1, batch_init_positions.shape[-1]).shape)
#             # print(batch_true_positions.shape, batch_true_positions.view(-1, batch_init_positions.shape[-1]).shape)
#             # print(out_images.shape, batch_true_images.shape)
#             loss += loss_fn(out_images[:], batch_true_images[:])
#             # .view(-1,batch_init_positions.shape[-1])
#         loss /= batch_size
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # update the progress bar
#         iterator.set_postfix_str(f'Loss: {loss.item():.8f}')
#         running_loss += loss.item()

#         if i % display == 0:
#            display_results_fn(i, model, out_display, getter, getter.total_length, getter.dt)
#            iterator.set_description_str(f'Display loss: {running_loss/display:.8f}')
#            running_loss = 0.


#         scheduler.step()
        
#     return None