import os, tqdm, argparse, imageio, clip
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

from datasets import LLFF_Dataset, LF5x5_Dataset, infiniteloop
from networks import VIINTER
from utils import linterp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='exp', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dset', default='LF', type=str)
    parser.add_argument('--scene', default='knights', type=str)
    parser.add_argument('--size', help='if not None, resize image', default=None, type=int)

    parser.add_argument('--start_lr', default=1e-5, type=float)
    parser.add_argument('--final_lr', default=1e-6, type=float)
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--z_dim', default=128, type=int)
    parser.add_argument('--clip', default=0.01, type=float)
    parser.add_argument('--percep_freq', help='compute clip loss every n-th iterations', default=2, type=int)

    parser.add_argument('--W', default=512, type=int)
    parser.add_argument('--D', default=8, type=int)
    parser.add_argument('--bsize', default=8192, type=int)
    parser.add_argument('--iters', default=300000, type=int)
    parser.add_argument('--save_freq', default=20000, type=int)
    parser.add_argument('--silent', action='store_true')

    args = parser.parse_args()

    exp_dir = f'{ROOT}/exps/{args.dset}/{args.scene}/{args.exp_name}'

    if args.dset == 'LF':
        dset = LF5x5_Dataset(f'{args.data_dir}/{args.scene}', size=args.size)

        val_start = 0
        val_end = 24

    elif args.dset == 'LLFF':
        dset = LLFF_Dataset(f'{args.data_dir}/{args.scene}', size=args.size)

        val_start = 0
        val_end = len(dset) - 1
        if args.scene == 'trex':
            val_start = 7
            val_end = 14
        if args.scene == 'room':
            val_start = 10
            val_end = 15
        if args.scene == 'fern':
            val_start = 0
            val_end = 4
        if args.scene == 'fortress':
            val_start = 0
            val_end = 5
        if args.scene == 'flower':
            val_start = 7
            val_end = 13
        if args.scene == 'horns':
            val_start = 0
            val_end = 7
        if args.scene == 'leaves':
            val_start = 9
            val_end = 14
        if args.scene == 'orchids':
            val_start = 0
            val_end = 3

    else:
        print("Dataset not supported")
        exit()

    if args.clip != 0.0:
        exp_dir += f'_clip_{args.clip}'
    exp_dir += f'_dim{args.z_dim}'
    exp_dir += f'_W{args.W}'
    exp_dir += f'_D{args.D}'

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f'{exp_dir}/val_out', exist_ok=True)
    os.makedirs(f'{exp_dir}/val_out/final_frames', exist_ok=True)

    bsize = args.bsize; num_steps = args.iters; save_freq = 1000
    inter_fn = linterp
    if args.p == 0: args.p = None

    net = VIINTER(n_emb = len(dset), norm_p = args.p, inter_fn=inter_fn, D=args.D, z_dim = args.z_dim, in_feat=2, out_feat=3, W=args.W, with_res=False, with_norm=True)
    net = net.to(DEVICE)
    train_loader = DataLoader(dset, batch_size=1, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.start_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=args.final_lr)

    min_side = min(dset.hw[0], dset.hw[1])
    ds_ratio = min_side // 256
    if args.clip > 0.0:
        clip_model, preprocess = clip.load("ViT-B/32")
        clip_model = clip_model.to(DEVICE).eval()
        clip_im_size = 224
        clip_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((clip_im_size, clip_im_size)),
                    torchvision.transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073), 
                        std=(0.26862954, 0.26130258, 0.27577711))]
                    )

        im_feats = []
        imgs = []
        print("Precomputing CLIP faatures")
        with torch.no_grad():
            for im in dset.imgs:
                if min_side > 256:
                    im = torchvision.transforms.Resize(min_side//ds_ratio)(im)
                im = im.to(DEVICE).unsqueeze(0)
                patches = F.unfold(im, kernel_size = (clip_im_size, clip_im_size), stride=clip_im_size)
                patches = patches.reshape(3, clip_im_size, clip_im_size, -1).permute(3, 0, 1, 2)
                im_feats.append(clip_model.encode_image(clip_transform(patches)).float().detach().squeeze())
                imgs.append(im.to(DEVICE))
        im_feats = torch.stack(im_feats, dim = 0)
        imgs = torch.stack(imgs, dim = 0)

    mse_losses, psnrs = [], []
    print(f'Starts training {exp_dir}')
    print(f'Image size: {dset.hw}')

    coords_h = np.linspace(-1, 1, dset.hw[0], endpoint=False)
    coords_w = np.linspace(-1, 1, dset.hw[1], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
    grid_inp = torch.FloatTensor(xy_grid).view(-1, 2).contiguous().unsqueeze(0).to(DEVICE)

    if min_side > 256:
        coords_h_ds = np.linspace(-1, 1, dset.hw[0]//ds_ratio, endpoint=False)
        coords_w_ds = np.linspace(-1, 1, dset.hw[1]//ds_ratio, endpoint=False)
        xy_grid_ds = np.stack(np.meshgrid(coords_w_ds, coords_h_ds), -1)
        grid_inp_ds = torch.FloatTensor(xy_grid_ds).view(-1, 2).contiguous().unsqueeze(0).to(DEVICE)

        clip_grid_inp = grid_inp_ds
        clip_hw = [dset.hw[0]//ds_ratio, dset.hw[1]//ds_ratio]

    else:
        clip_grid_inp = grid_inp
        clip_hw = [dset.hw[0], dset.hw[1]]

    test_psnrs, test_ssims = [], []

    steps = 0
    loop = tqdm.trange(num_steps, disable=args.silent)
    train_loader = infiniteloop(train_loader)
    for i in loop:
        img, ind = next(train_loader)

        optimizer.zero_grad()
        if args.clip != 0.0:
            if i % args.percep_freq == 0:
                mix_out, ia, ib, alpha, z = net.mix_forward(clip_grid_inp, batch_size=1)
                mix_out = mix_out.view(-1, clip_hw[0], clip_hw[1], 3)

                patches = F.unfold(mix_out.permute(0, 3, 1, 2), kernel_size = (clip_im_size, clip_im_size), stride=clip_im_size)
                patches = patches.reshape(3, clip_im_size, clip_im_size, -1).permute(3, 0, 1, 2)
                out_emb = clip_model.encode_image(clip_transform(patches)).float().squeeze()
                mix_emb = (im_feats[ia] * (1 - alpha)) + (im_feats[ib] * alpha)
                feats_loss = args.clip * F.mse_loss(out_emb, mix_emb.squeeze())
                feats_loss.backward()
        optimizer.step()

        num_pixels = img.shape[-2] * img.shape[-1]
        sind = torch.randperm(num_pixels)[:bsize].squeeze()
        img, ind = img[0].permute(1, 2, 0).reshape(-1, 3)[sind].to(DEVICE), torch.LongTensor([ind[0]]).to(DEVICE)

        optimizer.zero_grad()
        out = net(grid_inp[:, sind], ind).squeeze()
        mse_loss = F.mse_loss(out, img)
        loss = mse_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        psnr = 10 * np.log10(1 / mse_loss.item())
        steps += 1; loop.set_postfix(PSNR = psnr)

        if steps % args.save_freq == 0:
            if args.clip > 0.0:
                generated = torch.clamp(mix_out[0].detach().cpu(), 0, 1).numpy()
                Image.fromarray(np.uint8(255 * generated)).save(f'{exp_dir}/val_out/{steps}_mix.jpg')
            torch.save(net.state_dict(), f'{exp_dir}/net.pth')
            net.eval()
            with torch.no_grad():
                out = torch.zeros((grid_inp.shape[-2], 3))
                _b = 8192 * 4
                for ib in range(0, len(out), _b):
                    out[ib:ib+_b] = net(grid_inp[:, ib:ib+_b], torch.LongTensor([0]).to(DEVICE)).cpu()
            net.train()
            generated = torch.clamp(out.view(dset.hw[0], dset.hw[1], 3), 0, 1).numpy()
            Image.fromarray(np.uint8(255 * generated)).save(f'{exp_dir}/val_out/{steps}.jpg')

            frames_out = []
            with torch.no_grad():
                z0 = net.ret_z(torch.LongTensor([val_start]).to(DEVICE)).squeeze()
                z1 = net.ret_z(torch.LongTensor([val_end]).to(DEVICE)).squeeze()
                lin_sample_num = 30
                for a in torch.linspace(0, 1, lin_sample_num):
                    zi = inter_fn(a, z0, z1).unsqueeze(0)
                    out = torch.zeros((grid_inp.shape[-2], 3))
                    _b = 8192 * 4
                    for ib in range(0, len(out), _b):
                        out[ib:ib+_b] = net.forward_with_z(grid_inp[:, ib:ib+_b].to(DEVICE), zi).cpu()
                    generated = torch.clamp(out.view(dset.hw[0], dset.hw[1], 3), 0, 1).numpy()
                    frames_out.append(np.uint8(255 * np.clip(generated, 0, 1)))
            imageio.mimsave(f'{exp_dir}/val_out/{steps}.gif', frames_out, fps=10)

    torch.save(net.state_dict(), f'{exp_dir}/net.pth')

    for i, f in enumerate(frames_out):
        imageio.imsave(f'{exp_dir}/val_out/final_frames/{i}.png', f)

    training_psnr, training_ssim = 0, 0
    for i in range(len(dset)):
        with torch.no_grad():
            out = torch.zeros((grid_inp.shape[-2], 3))
            _b = 8192 * 8
            for ib in range(0, len(out), _b):
                out[ib:ib+_b] = net(grid_inp[:, ib:ib+_b].to(DEVICE), torch.LongTensor([i]).to(DEVICE)).cpu()
            generated = torch.clamp(out.view(dset.hw[0], dset.hw[1], 3), 0, 1)
            out = np.uint8(255 * np.clip(generated.numpy(), 0, 1))
            training_mse = F.mse_loss(dset.imgs[i].permute(1, 2, 0), generated).item()
            training_psnr += 10 * np.log10(1 / training_mse)
            training_ssim += ssim(np.clip(generated.numpy(), 0, 1), dset.imgs[i].permute(1, 2, 0).numpy(), channel_axis=2, multichannel=True)
    training_psnr, training_ssim = [training_psnr / len(dset)], [training_ssim / len(dset)]
    print(f'Training set | PSNR: {training_psnr[0]} | SSIM: {training_ssim[0]}')