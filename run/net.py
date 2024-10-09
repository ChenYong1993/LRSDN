from utils import *
import torch
from dgsan import DGSAN
from numpy import random


def load_model(index, input_depth, out_depth, g_depth):
    if index < 8:
        if index % 4 == 2:
            model, optimizer, loss_fn = model_load(input_depth, out_depth, g_depth, model_1='none')
        elif index % 4 == 3:
            model_S = torch.load('../model/model.pth').type(torch.FloatTensor).cuda()
            model, optimizer, loss_fn = model_load(input_depth, out_depth, g_depth, model_1=model_S)
        else:
            model, optimizer, loss_fn = model_load(input_depth, out_depth, g_depth, model_1='none')
    elif index < 19:
        if index % 4 == 1:
            model, optimizer, loss_fn = model_load(input_depth, out_depth, g_depth, model_1='none')
        elif index % 4 == 2:
            model_S = torch.load('../model/model.pth').type(torch.FloatTensor).cuda()
            model, optimizer, loss_fn = model_load(input_depth, out_depth, g_depth, model_1=model_S)
        elif index % 4 == 3:
            model_S = torch.load('../model/model.pth').type(torch.FloatTensor).cuda()
            model, optimizer, loss_fn = model_load(input_depth, out_depth, g_depth, model_1=model_S)
        else:
            model, optimizer, loss_fn = model_load(input_depth, out_depth, g_depth, model_1='none')
    else:
        model_S = torch.load('../model/model.pth').type(torch.FloatTensor).cuda()
        model, optimizer, loss_fn = model_load(input_depth, out_depth, g_depth, model_1=model_S)
    return model, optimizer, loss_fn


def model_load(input_depth, out_depth, g_depth, model_1='none'):
    if model_1 == 'none':
        model = DGSAN(input_channels=input_depth, output_channels=out_depth, guide_channels=g_depth, featrues_channels=64).type(torch.FloatTensor).cuda()
    else:
        model = model_1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss().cuda()
    return model, optimizer, loss_fn


def get_noise(data_size, noise_type='u', var=1. / 10):
    shape = [1, data_size[2], data_size[0], data_size[1]]
    net_input = torch.zeros(shape)
    if noise_type == 'u':
        net_input = net_input.uniform_() * var
    elif noise_type == 'n':
        net_input = net_input.normal_() * var
    else:
        assert False
    return net_input.cuda().float()


def net(truth_tensor, net_input, Phi_tensor, y_tensor, model, optimizer, loss_fn, guide_torch, net_iter, e_torch,
        g_tensor, lam, mu):
    loss_min = torch.tensor([100]).cuda().float()
    for i in range(net_iter):

        model_out = model(guide_torch, net_input)
        optimizer.zero_grad()
        out_x = torch.matmul(model_out.transpose(1, 2).transpose(2, 3), e_torch)
        out = out_x.transpose(2, 3).transpose(1, 2)
        y_hat = A_torch(out, Phi_tensor)
        y_loss = loss_fn(y_hat, y_tensor)
        g_loss = loss_fn(g_tensor.transpose(2, 3).transpose(1, 2), out)

        loss = y_loss * lam + g_loss * mu
        if (i + 1) % 25 == 0 and y_loss < 1.02 * loss_min:
            loss_min = y_loss
            output = model_out
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0 and y_loss < 1.02 * loss_min:
            torch.save(model, '../model/model.pth')
            PSNR = psnr_torch(truth_tensor, torch.squeeze(out_x).transpose(1, 2).transpose(0, 1))
            print('net_iter {}, y_loss:{:.7f}, g_loss:{:.7f}, PSNR:{:.3f}'.format(i + 1, y_loss.detach().cpu().numpy(),
                                                                                  g_loss.detach().cpu().numpy(),
                                                                                  PSNR.detach().cpu().numpy()))
    return output, loss_min.detach().cpu().numpy()
