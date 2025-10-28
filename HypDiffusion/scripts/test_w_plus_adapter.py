import os.path as osp
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
import re
import torch
import PIL
import argparse
import os
import sys
sys.path.insert(0, '/data2/mhf/DXL/Lingxiao/Codes/HypDiffusion')
from script.utils_direction import *
import w_plus_adapter
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import cv2
import time
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import numpy as np



def load_img(path, SCALE, pad=False, seg=False, target_size=None):
    if seg:
        # Load the input image and segmentation map
        image = Image.open(path).convert("RGB")
        seg_map = Image.open(seg).convert("1")

        # Get the width and height of the original image
        w, h = image.size

        # Calculate the aspect ratio of the original image
        aspect_ratio = h / w

        # Determine the new dimensions for resizing the image while maintaining aspect ratio
        if aspect_ratio > 1:
            new_w = int(SCALE * 256 / aspect_ratio)
            new_h = int(SCALE * 256)
        else:
            new_w = int(SCALE * 256)
            new_h = int(SCALE * 256 * aspect_ratio)

        # Resize the image and the segmentation map to the new dimensions
        image_resize = image.resize((new_w, new_h))
        segmentation_map_resize = cv2.resize(np.array(seg_map).astype(
            np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad the segmentation map to match the target size
        padded_segmentation_map = np.zeros((target_size[1], target_size[0]))
        start_x = (target_size[1] - segmentation_map_resize.shape[0]) // 2
        start_y = (target_size[0] - segmentation_map_resize.shape[1]) // 2
        padded_segmentation_map[start_x: start_x + segmentation_map_resize.shape[0],
                                start_y: start_y + segmentation_map_resize.shape[1]] = segmentation_map_resize

        # Create a new RGB image with the target size and place the resized image in the center
        padded_image = Image.new("RGB", target_size)
        start_x = (target_size[0] - image_resize.width) // 2
        start_y = (target_size[1] - image_resize.height) // 2
        padded_image.paste(image_resize, (start_x, start_y))

        # Update the variable "image" to contain the final padded image
        image = padded_image
    else:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        print(f"loaded input image of size ({w}, {h}) from {path}")
        # resize to integer multiple of 64
        w, h = map(lambda x: x - x % 64, (w, h))
        w = h = 512
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)

    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    if pad or seg:
        return 2. * image - 1., new_w, new_h, padded_segmentation_map

    return 2. * image - 1., w, h


def load_model_and_get_prompt_embedding(model, opt, device, prompts, inv=False):

    if inv:
        inv_emb = model.get_learned_conditioning(prompts, inv)
        c = uc = inv_emb
    else:
        inv_emb = None

    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(opt.n_samples * [""])
    else:
        uc = None
    c = model.get_learned_conditioning(prompts)

    return c, uc, inv_emb


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=gpu)
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

    # model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of a doggy, ultra realistic",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--ref-img",
        type=list,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--seg",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )

    parser.add_argument(
        "--dpm_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
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
        default=16,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=2.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="/data2/mhf/DXL/Lingxiao/Cache/model_weight/anydoor/v2-1_512-ema-pruned.ckpt",
        help="path to checkpoint of model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--root",
        type=str,
        help="",
        default='./inputs/cross_domain'
    )

    parser.add_argument(
        "--domain",
        type=str,
        help="",
        default='cross'
    )

    parser.add_argument(
        "--dpm_order",
        type=int,
        help="",
        choices=[1, 2, 3],
        default=2
    )

    parser.add_argument(
        "--tau_a",
        type=float,
        help="",
        default=0.4
    )

    parser.add_argument(
        "--tau_b",
        type=float,
        help="",
        default=0.8
    )

    parser.add_argument(
        "--gpu",
        type=str,
        help="",
        default='cuda:0'
    )

    opt = parser.parse_args()
    device = torch.device(
        opt.gpu) if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # The scale used in the paper
    if opt.domain == 'cross':
        opt.scale = 5.0
        file_name = "cross_domain"
    elif opt.domain == 'same':
        opt.scale = 2.5
        file_name = "same_domain"
    else:
        raise ValueError("Invalid domain")

    batch_size = opt.n_samples
    sample_path = os.path.join(outpath, file_name)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, opt.gpu)
    model = model.to(device)
    sampler = DPMSolverSampler(model)
    
    '''
    model parameter settings
    '''


    base_model_path = "/data2/mhf/DXL/Lingxiao/Cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5"
    # base_model_path = 'dreamlike-art/dreamlike-anime-1.0' #animate model
    # base_model_path = 'darkstorm2150/Protogen_x3.4_Official_Release'

    vae_model_path = "/data2/mhf/DXL/Lingxiao/Cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse"
    device = "cuda"
    wp_ckpt = '/data2/mhf/DXL/Lingxiao/Codes/w-plus-adapter/experiments_stage1_2024-07-30/checkpoint-34000/wplus_adapter.bin'

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    if not osp.exists('/data2/mhf/DXL/Lingxiao/Codes/w-plus-adapter/experiments_stage1_2024-07-30/checkpoint-34000/wplus_adapter.bin'):
        download_url = 'https://github.com/csxmli2016/w-plus-adapter/releases/download/v1/wplus_adapter_stage1.bin'
        load_file_from_url(
            url=download_url, model_dir='./pretrain_models/', progress=True, file_name=None)

    wp_model = w_plus_adapter.WPlusAdapter(pipe, wp_ckpt, device)

    for subdir, _, files in os.walk(opt.root):
        for file in files:
            torch.cuda.empty_cache()
            file_path = os.path.join(subdir, file)
            result = re.search(r'./inputs/[^/]+/(.+)/bg\d+\.', file_path)
            if result:
                prompt = result.group(1)

            if file_path.endswith('.jpg') or file_path.endswith('.jpeg') or file_path.endswith('.png'):
                if file.startswith('bg'):
                    opt.init_img = file_path
                elif file.startswith('fg') and not (file.endswith('mask.jpg') or file.endswith('mask.png')):
                    opt.ref_img = file_path
                elif file.startswith('mask'):
                    opt.mask = file_path
                elif file.startswith('fg') and (file.endswith('mask.jpg') or file.endswith('mask.png')):
                    opt.seg = file_path

            if file == files[-1]:
                seed_everything(opt.seed)
                img = cv2.imread(opt.mask, 0)
                # Threshold the image to create binary image
                _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                # Find the contours of the white region in the image
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Find the bounding rectangle of the largest contour
                x, y, new_w, new_h = cv2.boundingRect(contours[0])
                # Calculate the center of the rectangle
                center_x = x + new_w / 2
                center_y = y + new_h / 2
                # Calculate the percentage from the top and left
                center_row_from_top = round(center_y / 512, 2)
                center_col_from_left = round(center_x / 512, 2)

                aspect_ratio = new_h / new_w

                if aspect_ratio > 1:
                    scale = new_w * aspect_ratio / 256
                    scale = new_h / 256
                else:
                    scale = new_w / 256
                    scale = new_h / (aspect_ratio * 256)

                scale = round(scale, 2)

                # =============================================================================================

                assert prompt is not None
                data = [batch_size * [prompt]]

                # read background image
                assert os.path.isfile(opt.init_img)
                init_image, target_width, target_height = load_img(
                    opt.init_img, scale)
                init_image = repeat(init_image.to(device),
                                    '1 ... -> b ...', b=batch_size)
                save_image = init_image.clone()

                # read foreground image and its segmentation map
                ref_image, width, height, segmentation_map = load_img(
                    opt.ref_img, scale, seg=opt.seg, target_size=(target_width, target_height))
                ref_image = repeat(ref_image.to(device),
                                   '1 ... -> b ...', b=batch_size)

                segmentation_map_orig = repeat(torch.tensor(segmentation_map)[
                                               None, None, ...].to(device), '1 1 ... -> b 4 ...', b=batch_size)
                segmentation_map = segmentation_map_orig[:, :, ::8, ::8].to(
                    device)

                precision_scope = autocast if opt.precision == "autocast" else nullcontext

                # image composition
                with torch.no_grad():
                    with precision_scope("cuda"):
                        for prompts in data:
                            print(prompts)
                            c, uc, inv_emb = load_model_and_get_prompt_embedding(
                                model, opt, device, prompts, inv=True)

                            if opt.domain == 'same':  # same domain
                                init_image = save_image

                            T1 = time.time()
                            init_latent = model.get_first_stage_encoding(
                                model.encode_first_stage(init_image))

                            shape = [init_latent.shape[1],
                                     init_latent.shape[2], init_latent.shape[3]]

                            z_enc, _ = sampler.sample(steps=opt.dpm_steps,
                                                      inv_emb=inv_emb,
                                                      unconditional_conditioning=uc,
                                                      conditioning=c,
                                                      batch_size=opt.n_samples,
                                                      shape=shape,
                                                      verbose=False,
                                                      unconditional_guidance_scale=opt.scale,
                                                      eta=opt.ddim_eta,
                                                      order=opt.dpm_order,
                                                      x_T=init_latent,
                                                      width=width,
                                                      height=height,
                                                      DPMencode=True,
                                                      )

                            samples_orig = z_enc.clone()

                            samples_bg, _ = sampler.sample(steps=opt.dpm_steps,
                                                           inv_emb=inv_emb,
                                                           conditioning=c,
                                                           batch_size=opt.n_samples,
                                                           shape=shape,
                                                           verbose=False,
                                                           unconditional_guidance_scale=opt.scale,
                                                           unconditional_conditioning=uc,
                                                           eta=opt.ddim_eta,
                                                           order=opt.dpm_order,
                                                           x_T=[
                                                               samples_orig, init_latent],
                                                           width=width,
                                                           height=height,
                                                           segmentation_map=segmentation_map,
                                                           target_height=target_height,
                                                           target_width=target_width,
                                                           center_row_rm=center_row_from_top,
                                                           center_col_rm=center_col_from_left,
                                                           tau_a=opt.tau_a,
                                                           tau_b=opt.tau_b,
                                                           )

                            x_samples_bg = model.decode_first_stage(samples_bg)
                            x_samples_bg = torch.clamp(
                                (x_samples_bg + 1.0) / 2.0, min=0.0, max=1.0)

                            T2 = time.time()
                            print('Running Time: %s s' % ((T2 - T1)))

                            for x_sample in x_samples_bg:
                                x_sample = 255. * \
                                    rearrange(x_sample.cpu().numpy(),
                                              'c h w -> h w c')
                                img = Image.fromarray(
                                    x_sample.astype(np.uint8))
                                img.save(os.path.join(
                                    sample_path, f"{base_count:05}_reconstruction.png"))
                                base_count += 1

                del z_enc, samples_orig, x_sample, img, c, uc, inv_emb
                del init_image, init_latent, save_image, ref_image, prompt, prompts, data, binary, contours

    print(
        f"Your samples are ready and waiting for you here: \n{sample_path} \nEnjoy.")


if __name__ == "__main__":
    main()
