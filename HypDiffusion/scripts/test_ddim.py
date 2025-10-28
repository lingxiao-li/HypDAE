import numpy as np
import re
import torch
import PIL
import argparse
import os
import sys
sys.path.insert(0, '/data2/mhf/DXL/Lingxiao/Codes/HypDiffusion')
from ldm.models.diffusion.dpm_solver_org import DPMSolverSampler
from ldm.models.diffusion.ddim_org import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import time
import cv2
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from torch import autocast
from einops import rearrange, repeat
from itertools import islice
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from torchvision.utils import make_grid




def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path, size=[256, 256]):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h = image.shape[:2]
    print(f"loaded input image of size ({w}, {h}) from {path}")
    # resize to integer multiple of 32
    # w, h = map(lambda x: x - x % 32, (w, h))
    w, h = size
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    image = np.array(image).astype(np.uint8)

    image = (image / 127.5 - 1.0).astype(np.float32)
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def load_model_and_get_prompt_embedding(model, opt, device, prompts, sample_mode=True, inv=False):

    if inv:
        inv_emb = model.get_learned_conditioning(prompts, sample_mode=sample_mode, inv=inv)
        c = uc = inv_emb
    else:
        inv_emb = None

    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(opt.n_samples * [torch.zeros((1, 3, 224, 224))], sample_mode=sample_mode)
    else:
        uc = None
    c = model.get_learned_conditioning(prompts, sample_mode=sample_mode)

    return c, uc, inv_emb


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    
    parser.add_argument(
        "--ref-img",
        type=str,
        nargs="?",
        help="path to the reference image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )


    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )


    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device(
      "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # load init image
    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img, [512, 512]).to(device)
    init_image_resized = load_img(opt.init_img, [224, 224]).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_image_resized = repeat(init_image_resized, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(init_image))  # move to latent space
    
    # load ref image
    assert os.path.isfile(opt.ref_img)
    ref_image = load_img(opt.ref_img, [224, 224]).to(device)
    ref_image = repeat(ref_image, '1 ... -> b ...', b=batch_size)

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps,
                          ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                # c = model.get_learned_conditioning(prompts)
                c, uc, inv_emb = load_model_and_get_prompt_embedding(
                    model, opt, device, ref_image, inv=False)
                c_1, uc_1, inv_emb_1 = load_model_and_get_prompt_embedding(
                    model, opt, device, init_image_resized, inv=False)
                c_avg = (c + c_1) / 2
                print(f"conditioning shape: {c.shape}")
                shape = [opt.C, 64, 64]
                # encode (scaled latent)
                if opt.strength < 1.0:
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                # print(f"z_enc shape: {z_enc.shape}")
                else:
                    z_enc = torch.randn([opt.n_samples, 4, 64, 64], device=device)
                # decode it
                
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c_avg,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=c,
                                                    eta=opt.ddim_eta,
                                                    x_T=z_enc)
                '''
                samples_ddim = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                         unconditional_conditioning=uc,)
                                         '''

                x_samples = model.decode_first_stage(samples_ddim)
                x_samples = torch.clamp(
                    (x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                if not opt.skip_save:
                    for x_sample in x_samples:
                        x_sample = 255. * \
                            rearrange(x_sample.cpu().numpy(),
                                        'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                all_samples.append(x_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * \
                        rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(
                        os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
