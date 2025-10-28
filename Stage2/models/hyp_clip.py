"""
This file defines the core research contribution
"""
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from configs.paths_config import model_paths
from models.hyper_nets import MobiusLinear, HyperbolicMLR, LogisticRegression
import torch.nn.functional as F
from torch import nn
import torch
import geoopt.manifolds.stereographic.math as gmath
import math
from audioop import bias
import matplotlib
from .xf import LayerNorm, Transformer
matplotlib.use('Agg')


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


class hae_clip(nn.Module):

    def __init__(self, opts):
        super(hae_clip, self).__init__()
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
            self.num_classes = 2374
        elif self.opts.dataset_type == 'nabirds_encode':
            self.num_classes = 555
        elif self.opts.dataset_type == 'nabirds_encode_eva':
            self.num_classes = 444
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
            self.encoder.load_state_dict(
                get_keys(ckpt, 'encoder'), strict=True)
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

    def forward(self, x, y=None, batch_size=4, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, input_feature=False):
        if not input_feature:
            if input_code:
                codes = x
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
            feature_dist = x
            # codes = self.encoder(y)

        if self.opts.hyperbolic:
            logits = F.log_softmax(self.hyp_mlr(
                feature_dist, self.hyp_mlr.c), dim=-1)
            feature_euc = gmath.logmap0(feature_dist, k=torch.tensor(-float(self.hyp_mlr.c)))
        else:
            # feature_euc = feature_dist
            logits = F.log_softmax(
                self.euc_mlr(feature_dist), dim=-1)
            feature_euc = feature_dist

        feature_euc = feature_euc.unsqueeze(1)
        feature_euc = self.linear_decoder(feature_euc)
        feature_euc = self.transformer_decoder(feature_euc)
        codes = feature_euc


        return logits, feature_dist, ocodes, feature_euc

    def set_opts(self, opts):
        self.opts = opts
