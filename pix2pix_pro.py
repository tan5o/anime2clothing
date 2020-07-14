import torch
from generator.networks import define_G, define_D
from discriminator.discriminator_model import PG_MultiScaleDiscriminator, PG_MultiPatchDiscriminator
from generator.networks import GANLoss, get_scheduler, update_learning_rate

from generator.base_model import BaseModel

class Pix2PixPro(BaseModel):
    def name(self):
        return 'Pix2PixPro'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.net_g = define_G(netG=opt.netG, gpu_id=opt.gpu_ids, param_rate=opt.param_rate)
        self.net_d = define_D(netD="multi_scale", gpu_id=opt.gpu_ids)
        self.net_d_tf = define_D(netD="multi_patch", gpu_id=opt.gpu_ids)


        self.criterionGAN = GANLoss(gan_mode=opt.gan_mode, multi_scale=True)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionFM = torch.nn.L1Loss()

        self.l1_lambda = 10
        self.fm_lambdas = [5.0, 1.5, 1.5, 1.5, 1.0]

        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_d_tf = torch.optim.Adam(self.net_d_tf.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.net_g_scheduler = get_scheduler(self.optimizer_g, opt)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, opt)
        self.net_d_tf_scheduler = get_scheduler(self.optimizer_d_tf, opt)

        self.gpu_ids = opt.gpu_ids

        # load networks
        if opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            if opt.load_epoch == 0:
                load_epoch = "latest"
            else:
                load_epoch = opt.load_epoch

            print("Load model at: {}".format(load_epoch))

            self.load_network(self.net_g, 'G', load_epoch, pretrained_path)
            self.load_network(self.net_d, 'D', load_epoch, pretrained_path)
            self.load_network(self.net_d_tf, 'D_tf', load_epoch, pretrained_path)

            self.load_network(self.optimizer_g, 'G_optim', load_epoch, pretrained_path, is_optimizer=True)
            self.load_network(self.optimizer_d, 'D_optim', load_epoch, pretrained_path, is_optimizer=True)
            self.load_network(self.optimizer_d_tf, 'D_optim_tf', load_epoch, pretrained_path, is_optimizer=True)

    def forward(self, real_256_a, real_a, real_b, other_b):
        ### Generate Fake Image ###
        fake_b = self.net_g(real_256_a.detach())
        losses = {}

        real_ab = torch.cat((real_a, real_b), 1)
        fake_ab = torch.cat((real_a, fake_b), 1)
        other_ab = torch.cat((real_a, other_b), 1)

        ### Domain Discriminator Loss ###
        pred_fake = self.net_d.forward(fake_ab.detach())
        losses["D_fake"] = self.criterionGAN(pred_fake, False, net_type="D")
        pred_real = self.net_d.forward(real_ab)
        losses["D_real"] = self.criterionGAN(pred_real, True, net_type="D")
        pred_other = self.net_d.forward(other_ab)
        losses["D_other"] = self.criterionGAN(pred_other, False, net_type="D")

        ### Real/Fake Discriminator Loss ###
        pred_fake = self.net_d_tf.forward(fake_b.detach())
        losses["D_tf_fake"] = self.criterionGAN(pred_fake, False, net_type="D")
        pred_real = self.net_d_tf.forward(real_b.detach())
        losses["D_tf_real"] = self.criterionGAN(pred_real, True, net_type="D")

        ### Generator Loss ###
        pred_fake = self.net_d.forward(fake_ab, pop_intermediate=True)
        losses["G_GAN"] = self.criterionGAN(pred_fake, True, net_type="G")
        ## FeatureMatch
        pred_real = self.net_d.forward(real_ab, pop_intermediate=True)
        losses["G_FM"] = 0

        for index in range(len(pred_fake)):
            for (fake_i, real_i, lam) in zip(pred_fake[index][:-1], pred_real[index][:-1], self.fm_lambdas):
                losses["G_FM"] += self.criterionFM(fake_i, real_i.detach()) * lam

        pred_fake_tf = self.net_d_tf.forward(fake_b, pop_intermediate=True)
        losses["G_GAN_tf"] = self.criterionGAN(pred_fake_tf[-1], True, net_type="G")

        if self.opt.no_ic_loss:
            pred_real_tf = self.net_d_tf.forward(real_b, pop_intermediate=True)
            losses["G_FM_tf"] = 0
            for index in range(len(pred_fake_tf)):
                for (fake_i, real_i, lam) in zip(pred_fake_tf[index][:-1], pred_real_tf[index][:-1], self.fm_lambdas):
                    losses["G_FM_tf"] += self.criterionFM(fake_i, real_i.detach()) * lam
        elif not self.opt.no_ic_loss:
            pred_real_a = self.net_d_tf.forward(real_a, pop_intermediate=True)
            losses["G_Input"] = 0
            for index in range(len(pred_fake_tf)):
                for (fake_i, real_i, lam) in zip(pred_fake_tf[index][:-1], pred_real_a[index][:-1], self.fm_lambdas):
                    losses["G_Input"] += self.criterionFM(fake_i, real_i.detach()) * lam

        losses["G_L1"] = self.criterionL1(fake_b, real_b) * self.l1_lambda

        return losses, fake_b

    def inference(self, real_256_a):
        with torch.no_grad():
            fake_b = self.net_g(real_256_a.detach())
        return fake_b

    def set_config(self, current_resolution, status, alpha):
        self.net_g.set_config(current_resolution, status, alpha)
        self.net_d.set_config(current_resolution, status, alpha)
        self.net_d_tf.set_config(current_resolution, status, alpha)

    def update_learning_rate(self):
        update_learning_rate(self.net_g_scheduler, self.optimizer_g)
        update_learning_rate(self.net_d_scheduler, self.optimizer_d)
        update_learning_rate(self.net_d_tf_scheduler, self.optimizer_d_tf)

    def save(self, which_epoch):
        print("Checkpoint saved: {}".format(which_epoch))

        self.save_network(self.net_g, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.net_d, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.net_d_tf, 'D_tf', which_epoch, self.gpu_ids)

        self.save_network(self.optimizer_g, 'G_optim', which_epoch, self.gpu_ids)
        self.save_network(self.optimizer_d, 'D_optim', which_epoch, self.gpu_ids)
        self.save_network(self.optimizer_d_tf, 'D_optim_tf', which_epoch, self.gpu_ids)


