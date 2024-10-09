import scipy.io as sio
import os
from numpy import linalg as la
from net import *
from utils import *
from numpy import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

lam = 0.1
mu = 0.03
name = 'balloons'
rank = 7
r_up = 3
# load data
datapath = '../data/' + name + '31_cassi.mat'  # data path
X_ori = sio.loadmat(datapath)['orig']
maskpath = '../data/dd_mask.mat'
Phi = sio.loadmat(maskpath)['mask']
gap_path = '../initial/' + name + '_gap.mat'
X_gap = sio.loadmat(gap_path)['rec']
X_ori = X_ori / X_ori.max()
y = A(X_ori, Phi)

Phi_tensor = torch.unsqueeze(torch.from_numpy(np.transpose(Phi, (2, 0, 1))), 0).cuda().float()
y_tensor = torch.unsqueeze(torch.from_numpy(y), 0).cuda().float()
truth_tensor = torch.from_numpy(np.transpose(X_ori, (2, 0, 1))).cuda().float()
noise_shape = [0, 0, 0]
for i in range(3):
    if i != 2:
        noise_shape[i] = int(X_ori.shape[i] / (2 ** 5))
    else:
        noise_shape[i] = int(X_ori.shape[i])
net_input = get_noise(noise_shape)
Phi_sum = np.sum(Phi ** 2, 2)
b = np.zeros([512, 512, 31])
x = X_gap
rec = x.transpose(2, 0, 1).reshape(X_ori.shape[2], X_ori.shape[0] * X_ori.shape[1])
u1, sigma, vt = la.svd(rec, full_matrices=False)
e = u1[:, 0:rank]
g_t = np.matmul(x, e[:, 0:rank])
guide_torch = torch.unsqueeze(torch.from_numpy(np.transpose(g_t, (2, 0, 1))), 0).cuda().float()
g_depth = rank

for i in range(24):
    if i % 4 == 0 and i != 0:
        if i < 17:
            rank = rank + r_up
        rec = x.transpose(2, 0, 1).reshape(X_ori.shape[2], X_ori.shape[0] * X_ori.shape[1])
        u1, sigma, vt = la.svd(rec, full_matrices=False)
        e = u1[:, 0:rank]

    iterl = 1000 + 100 * i
    g_tensor = torch.unsqueeze(torch.from_numpy(x - b), 0).cuda().float()
    e_torch = torch.from_numpy(e.transpose()).cuda().float()
    inp = net_input
    input_depth = inp.shape[1]
    out_depth = rank
    model, optimizer, loss_fn = load_model(i, input_depth, out_depth, g_depth)
    out, loss_y_iter = net(truth_tensor, inp, Phi_tensor, y_tensor, model, optimizer, loss_fn, guide_torch,
                           iterl, e_torch, g_tensor, lam, mu)
    x_t = np.squeeze(torch.matmul(out.transpose(1, 2).transpose(2, 3), e_torch).detach().cpu().numpy())
    c = x_t + b
    yb = A(c, Phi)
    x = c + At(np.divide(y - yb, Phi_sum + mu), Phi)
    b = b - (x - x_t)
    mu = mu * 1.1
    guide_torch = out.detach()
    g_depth = rank
    psnr_x = psnr_block(X_ori, x)
    print('iter{}：PSNR_x={:.3f}'.format(i + 1, psnr_x))
psnr_x = psnr_block(X_ori, x)
print('result：PSNR:{:.3f}'.format(psnr_x))

