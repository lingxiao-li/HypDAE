import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from .hyper_nets import MobiusLinear, HyperbolicMLR, LogisticRegression
import open_clip
from ldm.util import default, count_params
import einops
from .xf import LayerNorm, Transformer
import geoopt.manifolds.stereographic.math as gmath
import math
from audioop import bias
import matplotlib


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items()
              if k[:len(name)] == name}
    return d_filt


class EqualLinear_encoder(nn.Module):
    def __init__(
            self, in_dim, out_dim):
        super(EqualLinear_encoder, self).__init__()
        self.out_dim = out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        return out


class EqualLinear_decoder(nn.Module):
    def __init__(
            self, in_dim, out_dim):
        super(EqualLinear_decoder, self).__init__()
        self.out_dim = out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)
        self.fc3 = nn.Linear(in_dim, in_dim)

        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        return out


class transformer_encoder(nn.Module):
    def __init__(self, dim):
        super(transformer_encoder, self).__init__()
        self.encoder = EqualLinear_encoder(1024, dim)

    def forward(self, dw):
        x = self.encoder(dw)
        return x


class transformer_decoder(nn.Module):
    def __init__(self, dim):
        super(transformer_decoder, self).__init__()
        self.dim = dim
        self.decoder_low = EqualLinear_decoder(dim, 1024)

    def forward(self, dw):
        shape = dw[:, :self.dim].shape
        x0 = self.decoder_low(dw[:, :self.dim])
        x = x0.reshape((shape[0], 1024))
        return x

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, inv=False):
        tokens = open_clip.tokenize(text)
        if inv:          
            tokens[0] = torch.zeros(77) + 7788
            z = self.encode_with_transformer(tokens.to(self.device), inv)
        else:
            z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text, inv=False):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        if inv == False:
            # x = einops.repeat(x[:,0], 'i j -> i c j', c=77)
            x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text, inv=False, device=None):
        if device is not None:
            self.device = device
        return self(text, inv)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="/data2/mhf/DXL/Lingxiao/Cache/huggingface/hub/models--openai--clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
            1,
            1024,
            5,
            1,
        )

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, image, inv=False):
        if isinstance(image, list):
            image = torch.cat(image, 0)
        if inv:
            image_shape = image.shape
            image = torch.zeros(image_shape) + 0.5
        
        image = image
        outputs = self.transformer(pixel_values=image)
        z = outputs.pooler_output
        z = z.unsqueeze(1)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, image, inv=False, device=None):
        if device is not None:
            self.device = device
        return self(image, inv)
    
    
class FrozenCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="/data2/mhf/DXL/Lingxiao/Cache/huggingface/hub/models--openai--clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
            1,
            1024,
            5,
            1,
        )
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, text, inv=False):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        print(outputs.last_hidden_state.shape)
        print(outputs.pooler_output.shape)
        z = z.unsqueeze(1)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, text, inv=False, device=None):
        if device is not None:
            self.device = device
        return self(text, inv)
    
    
class HyperbolicCLIPImageEmbedder(AbstractEncoder):
    def __init__(self, opts):
        super(HyperbolicCLIPImageEmbedder, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.encoder = self.set_encoder(self.opts.encoder_version)
        self.feature_shape = self.opts.feature_size
        if self.opts.dataset_type == 'flowers_encode':
            self.num_classes = 102  # animal_faces 151, flowers 102
        elif self.opts.dataset_type == 'flowers_encode_eva':
            self.num_classes = 85
        elif self.opts.dataset_type == 'animalfaces_encode':
            self.num_classes = 151
        elif self.opts.dataset_type == 'animalfaces_encode_eva':
            self.num_classes = 121
        elif self.opts.dataset_type == 'vggfaces_encode':
            self.num_classes = 2374
        elif self.opts.dataset_type == 'vggfaces_encode_eva':
            self.num_classes = 1802
        elif self.opts.dataset_type == 'ffhq_encode':
            self.num_classes = 1
        elif self.opts.dataset_type == 'nabirds_encode':
            self.num_classes = 555
        else:
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        self.transformer_encoder = Transformer(
            1,
            1024,
            5,
            1,
        )

        self.linear_encoder = LogisticRegression(1024, self.feature_shape)
        self.linear_decoder = LogisticRegression(self.feature_shape, 1024)
        if self.opts.dataset_type == 'flowers_encode':
            self.transformer_decoder = Transformer(
                1,
                1024,
                5,
                1,
            )
        else:
            self.transformer_decoder = Transformer(
                1,
                1024,
                self.opts.transformer_layers,
                1,
            )
        self.linear_encoder = LogisticRegression(1024, self.feature_shape)
        self.linear_decoder = LogisticRegression(self.feature_shape, 1024)
        # self.mlp = MLP(512)
        print('Use hyperbolic:', self.opts.hyperbolic)
        if self.opts.hyperbolic:
            self.hyperbolic_linear = MobiusLinear(self.feature_shape,
                                                  self.feature_shape,
                                                  # This computes an exmap0 after the operation, where the linear
                                                  # operation operates in the Euclidean space.
                                                  hyperbolic_input=False,
                                                  hyperbolic_bias=True,
                                                  nonlin=None,  # For now
                                                  )
            self.hyp_mlr = HyperbolicMLR(
                ball_dim=self.feature_shape, n_classes=self.num_classes, c=1)
        else:
            self.linear = LogisticRegression(
                self.feature_shape, self.feature_shape)
            self.euc_mlr = LogisticRegression(
                n_inputs=self.feature_shape, n_outputs=self.num_classes)
        # Load weights if needed
        self.load_weights()

    def set_encoder(self, version):
        if self.opts.encoder_type == 'CLIPImageEmbedder':
            encoder = FrozenCLIPImageEmbedder(version)
        else:
            raise Exception('{} is not a valid encoders'.format(
                self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading HAE from checkpoint: {}'.format(
                self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path,
                              map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            ckpt['state_dict'] = new_state_dict
            if self.opts.hyperbolic:
                self.hyperbolic_linear.load_state_dict(
                    get_keys(ckpt, 'hyperbolic_linear'), strict=True)
                self.hyp_mlr.load_state_dict(
                    get_keys(ckpt, 'hyp_mlr'), strict=True)
            else:
                self.linear.load_state_dict(
                    get_keys(ckpt, 'linear'), strict=True)
                self.euc_mlr.load_state_dict(
                    get_keys(ckpt, 'euc_mlr'), strict=True)
            self.transformer_encoder.load_state_dict(
                get_keys(ckpt, 'transformer_encoder'), strict=True)
            self.transformer_decoder.load_state_dict(
                get_keys(ckpt, 'transformer_decoder'), strict=True)
            self.linear_encoder.load_state_dict(
                get_keys(ckpt, 'linear_encoder'), strict=True)
            self.linear_decoder.load_state_dict(
                get_keys(ckpt, 'linear_decoder'), strict=True)
            ckpt_clip = torch.load(self.opts.psp_checkpoint_path,
                                   map_location=torch.device('cpu'))
            new_state_dict_clip = OrderedDict()
            for k, v in ckpt_clip['state_dict'].items():
                name = k.replace('.module', '')
                new_state_dict_clip[name] = v
            ckpt_clip['state_dict'] = new_state_dict_clip
            self.encoder.load_state_dict(
                get_keys(ckpt_clip, 'cond_stage_model'), strict=True)
            # self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        else:
            print('Initializing hyperediting from scratch')
            ckpt = torch.load(self.opts.psp_checkpoint_path,
                              map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                name = k.replace('.module', '')
                new_state_dict[name] = v
            ckpt['state_dict'] = new_state_dict
            self.encoder.load_state_dict(
                get_keys(ckpt, 'cond_stage_model'), strict=True)

    def forward(self, x, y=None, batch_size=4, resize=True, latent_mask=None, input_code=False,
                inject_latent=None, return_latents=False, alpha=None, input_feature=False):
        if isinstance(x, list):
            x = torch.cat(x, 0)

        if not input_feature:
            if input_code:
                ocodes = None
                feature = x
            else:
                codes = self.encoder(x)
                ocodes = codes
                feature = self.transformer_encoder(ocodes)
                feature = self.linear_encoder(feature)
            feature_reshape = torch.flatten(feature, start_dim=1)
            if self.opts.hyperbolic:
                feature_dist = self.hyperbolic_linear(feature_reshape)
            else:
                feature_dist = self.linear(feature_reshape)

        else:
            ocodes = None
            feature = None
            feature_dist = x
            # codes = self.encoder(y)

        if self.opts.hyperbolic:
            logits = F.log_softmax(self.hyp_mlr(
                feature_dist, self.hyp_mlr.c), dim=-1)
            feature_euc = gmath.logmap0(feature_dist, k=torch.tensor(-1.))
        else:
            # feature_euc = feature_dist
            logits = F.log_softmax(
                self.euc_mlr(feature_dist), dim=-1)
            feature_euc = feature_dist

        feature_euc = feature_euc.unsqueeze(1)
        feature_euc = self.linear_decoder(feature_euc)
        feature_euc = self.transformer_decoder(feature_euc)
        codes = feature_euc

        return logits, ocodes, feature, feature_dist, feature_euc

    def set_opts(self, opts):
        self.opts = opts

    def encode(self, image, input_feature=False, input_code=False, device=None):
        if device is not None:
            self.device = device
        return self(image, input_feature=input_feature, input_code=input_code)


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)

