import sys
sys.path.append(".")
sys.path.append("..")

from training.ranger import Ranger
from models.hyp_clip import hae_clip
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from criteria.lpips.lpips import LPIPS
from datasets_hae.images_dataset import ImagesDataset
from configs import data_configs
from criteria import id_loss, w_norm, moco_loss, contrastive_loss
from utils import common, train_utils
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, autograd
import torch
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import warnings
import torch.nn.parallel
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn

matplotlib.use('Agg')

random.seed(0)
torch.manual_seed(0)


class Coach:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts
        self.global_step = 1
        # torch.cuda.set_device(1)
        # self.device = 'cuda:1'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
        # self.opts.device = self.device

        if self.opts.use_wandb:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)

        if self.opts.seed is not None:
            random.seed(self.opts.seed)
            torch.manual_seed(self.opts.seed)
            cudnn.deterministic = True
            warnings.warn(
                "You have chosen to seed training. "
                "This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints."
            )

        if self.opts.gpu is not None:
            warnings.warn(
                "You have chosen a specific GPU. This will completely "
                "disable data parallelism."
            )

        if self.opts.dist_url == "env://" and self.opts.world_size == -1:
            self.opts.world_size = int(os.environ["WORLD_SIZE"])

        self.opts.distributed = self.opts.world_size > 1 or self.opts.multiprocessing_distributed

        # ngpus_per_node = torch.cuda.device_count()
        if self.opts.ngpus_per_node > torch.cuda.device_count():
            raise ValueError(
                "Cannot use more GPUs than available. Please specify a valid number of GPUs."
            )
        elif self.opts.ngpus_per_node is not None:
            ngpus_per_node = self.opts.ngpus_per_node
        else:
            ngpus_per_node = torch.cuda.device_count()
        if self.opts.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            self.opts.world_size = ngpus_per_node * self.opts.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(self.train, nprocs=ngpus_per_node,
                     args=(5, ngpus_per_node, prev_train_checkpoint))
        else:
            # Simply call main_worker function
            self.train(self.opts.gpu, 5, ngpus_per_node, prev_train_checkpoint)
    
            
    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(
                ckpt['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(
                ckpt['discriminator_optimizer_state_dict'])
        print(f'Resuming training from step {self.global_step}')

    def train(self, gpu, cpu, ngpus_per_node, prev_train_checkpoint=None):
        self.opts.gpu = gpu
        self.opts.device = torch.device("cuda:{}".format(gpu))

        if self.opts.gpu is not None:
            print("Use GPU: {} for training".format(self.opts.gpu))

        if self.opts.distributed:
            if self.opts.dist_url == "env://" and self.opts.rank == -1:
                self.opts.rank = int(os.environ["RANK"])
            if self.opts.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.opts.rank = self.opts.rank * ngpus_per_node + self.opts.gpu
            dist.init_process_group(
                backend=self.opts.dist_backend,
                init_method=self.opts.dist_url,
                world_size=self.opts.world_size,
                rank=self.opts.rank,
            )
        # create model
        print("=> creating model '{}'".format('HAE_editing'))
        self.net = hae_clip(self.opts)

        if self.opts.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.opts.gpu is not None:
                torch.cuda.set_device(self.opts.gpu)
                self.net.cuda(self.opts.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.opts.batch_size = int(
                    self.opts.batch_size / ngpus_per_node)
                self.opts.workers = int(
                    (self.opts.workers + ngpus_per_node - 1) / ngpus_per_node)
                self.net = torch.nn.parallel.DistributedDataParallel(
                    self.net, device_ids=[self.opts.gpu]
                )
            else:
                self.net.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.net = torch.nn.parallel.DistributedDataParallel(self.net)
        elif self.opts.gpu is not None:
            torch.cuda.set_device(self.opts.gpu)
            self.net = self.net.cuda(self.opts.gpu)
            # comment out the following line for debugging
            raise NotImplementedError(
                "Only DistributedDataParallel is supported.")
        else:
            # AllGather implementation (batch shuffle, queue update, etc.) in
            # this code only supports DistributedDataParallel.
            raise NotImplementedError(
                "Only DistributedDataParallel is supported.")


        # Initialize loss
        if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
            raise ValueError(
                'Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')
        self.mse_loss = nn.MSELoss().to(self.opts.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(
                net_type='alex', device=self.opts.device).to(self.opts.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.opts.device).eval()
        if self.opts.moco_lambda > 0:
            self.moco_loss = moco_loss.MocoLoss().to(self.opts.device).eval()
        if self.opts.contrastive_lambda > 0:
            self.contrastive_loss = contrastive_loss.SupConLoss().to(self.opts.device).eval()

        # Initialize optimizer
        self.optimizer, self.scheduler = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        if self.opts.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset)
        else:
            train_sampler = None
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=(train_sampler is None),
                                           num_workers=int(self.opts.workers),
                                           sampler=train_sampler,
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=(train_sampler is None),
                                          num_workers=int(
                                              self.opts.test_workers),
                                          sampler=train_sampler,
                                          drop_last=True)
        

        # Initialize logger
        log_dir = os.path.join(self.opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(self.opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

        self.net.train()
            
        while self.global_step < self.opts.max_steps:
            correct = 0
            data_size = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                self.optimizer.zero_grad()
                x, y, label = batch
                x, y, label = x.to(self.opts.device).float(), y.to(
                    self.opts.device).float(), label.to(self.opts.device)
                # images = torch.cat([x, z], dim=0)
                images = x
                bsz = label.shape[0]
                data_size += bsz
                logits, feature_dist, ocodes, feature_euc = self.net.forward(
                    images, batch_size=bsz, return_latents=True)
                loss, encoder_loss_dict, id_logs = self.calc_loss(logits, label, ocodes, feature_euc)
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                loss_dict = {**loss_dict, **encoder_loss_dict}
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')
                    if self.global_step != 0 and data_size != 0:
                        print("Test set: Accuracy: {}/{} ({:.0f}%)\n".format(correct,
                              data_size, 100.0*correct/data_size))
                        correct = 0
                        data_size = 0


                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1


    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_optimizers(self):
        print('training all parameters' if self.opts.train_all else 'training only decoder parameters')
        if self.opts.train_all:
            params = list(self.net.module.transformer_encoder.parameters())
            params += list(self.net.module.transformer_decoder.parameters())
            params += list(self.net.module.linear_encoder.parameters())
            params += list(self.net.module.linear_decoder.parameters())
            if self.opts.hyperbolic:
                params += list(self.net.module.hyperbolic_linear.parameters())
                params += list(self.net.module.hyp_mlr.parameters())
            else:
                params += list(self.net.module.linear.parameters())
                params += list(self.net.module.euc_mlr.parameters())
        else:
            params = list(self.net.module.transformer_encoder.parameters())
            
        # params = list(self.net.module.transformer_decoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.AdamW(params, lr=self.opts.learning_rate)
            scheduler = StepLR(optimizer, step_size=self.opts.step_size, gamma=self.opts.gamma)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer, scheduler

    def configure_datasets(self):
        # print(transforms_dict)
        train_dataset = ImagesDataset(self.opts.dataset_type)
        test_dataset = ImagesDataset(self.opts.dataset_type)
        if self.opts.use_wandb:
            self.wb_logger.log_dataset_wandb(
                train_dataset, dataset_name="Train")
            self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def calc_loss(self, logits, label, ocodes, feature_euc):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.hyperbolic_lambda > 0:
            loss_hyperbolic = F.nll_loss(logits, label)
            loss_dict['loss_hyperbolic'] = float(loss_hyperbolic)
            loss += loss_hyperbolic * self.opts.hyperbolic_lambda
        if self.opts.reverse_lambda > 0:
            loss_reverse = F.mse_loss(ocodes, feature_euc)
            if float(loss_reverse) <= 0.2 and float(loss_reverse) > 0.1:
                reverse_lambda = 3
            elif float(loss_reverse) <= 0.1 and float(loss_reverse) > 0.05:
                reverse_lambda = 6
            elif float(loss_reverse) <= 0.05 and float(loss_reverse) > 0.025:
                reverse_lambda = 12
            elif float(loss_reverse) <= 0.025 and float(loss_reverse) > 0.0125:
                reverse_lambda = 24
            elif float(loss_reverse) <= 0.0125 and float(loss_reverse) > 0.00625:
                reverse_lambda = 48
            elif float(loss_reverse) <= 0.00625 and float(loss_reverse) > 0.003125:
                reverse_lambda = 96
            elif float(loss_reverse) <= 0.003125 and float(loss_reverse) > 0.002:
                reverse_lambda = 200
            elif float(loss_reverse) <= 0.002 and float(loss_reverse) > 0.001:
                reverse_lambda = 300
            elif float(loss_reverse) <= 0.001 and float(loss_reverse) > 0.0005:
                reverse_lambda = 500
            elif float(loss_reverse) <= 0.0005 and float(loss_reverse) > 0.0001:
                reverse_lambda = 1000
            elif float(loss_reverse) <= 0.0001 and float(loss_reverse) > 0.00005:
                reverse_lambda = 2000
            elif float(loss_reverse) <= 0.00005 and float(loss_reverse) > 0.000025:
                reverse_lambda = 4000
            else:
                reverse_lambda = self.opts.reverse_lambda
            loss_dict['loss_reverse'] = float(loss_reverse)
            loss += loss_reverse * reverse_lambda
        if self.opts.cosine_lambda > 0:
            loss_cosine = F.cosine_similarity(ocodes.squeeze(1), feature_euc.squeeze(1), dim=1)
            loss_cosine_mean = torch.mean(1-loss_cosine)
            loss_dict['loss_cosine'] = float(loss_cosine_mean)
            loss += loss_cosine_mean * reverse_lambda
        loss += 0. * sum(p.sum() for p in self.net.module.parameters())
        '''
		if self.opts.contrastive_lambda > 0:
			loss_contrastive = self.contrastive_loss(feature_dist)
			loss_dict['info_nce'] = float(loss_contrastive)
			loss += loss_contrastive * self.opts.contrastive_lambda
		'''

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name,
                                f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.module.state_dict(),
            'opts': vars(self.opts)
        }
        return save_dict


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(
            grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
