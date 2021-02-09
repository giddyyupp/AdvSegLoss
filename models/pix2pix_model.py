import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .segmentation.detectron2 import panoptic_segmenter


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        # parser.set_defaults(netG='unet_256')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--lambda_Seg', type=float, default=100.0, help='weight for seg discriminator')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.lambda_Seg = opt.lambda_Seg
        self.segmentation = opt.segmentation

        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, depth=18,
                                      fpn_weights=opt.fpn_weights)

        if self.segmentation:
            # detectron2 segmenter
            self.segModel = panoptic_segmenter.PanopticSegmenter(opt.segmentation_output)

        if self.isTrain:
            # initialize optimizers
            self.optimizers = []

            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

            if self.segmentation:
                if opt.segmentation_output == "binary":
                    self.fake_seg_pool_binary = ImagePool(opt.pool_size)
                    out_seg_cls = 3
                    self.netD_Seg = networks.define_D(out_seg_cls+opt.input_nc, opt.ndf, opt.netD,
                                                      opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type,
                                                      opt.init_gain,
                                                      self.gpu_ids)
                    self.loss_names.append('D_Seg')
                    self.model_names.append('D_Seg')

                    self.optimizer_D_Seg = torch.optim.Adam(itertools.chain(self.netD_Seg.parameters()),
                                                            lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_D_Seg)

                elif opt.segmentation_output == "multi_class":
                    self.fake_seg_pool_multi_class = ImagePool(opt.pool_size)
                    out_seg_cls = 135
                    self.netD_Seg2 = networks.define_D(out_seg_cls+opt.input_nc, opt.ndf, opt.netD,
                                                       opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type,
                                                       opt.init_gain,
                                                       self.gpu_ids)
                    self.loss_names.append('D_Seg2')
                    self.model_names.append('D_Seg2')

                    self.optimizer_D_Seg = torch.optim.Adam(itertools.chain(self.netD_Seg2.parameters()),
                                                            lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_D_Seg)

                else:
                    self.fake_seg_pool_binary = ImagePool(opt.pool_size)
                    self.fake_seg_pool_multi_class = ImagePool(opt.pool_size)

                    out_seg_cls_binary = 3
                    out_seg_cls_multi_class = 135
                    self.netD_Seg = networks.define_D(out_seg_cls_binary+opt.input_nc, opt.ndf, opt.netD,
                                                      opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                                      self.gpu_ids)
                    self.netD_Seg2 = networks.define_D(out_seg_cls_multi_class+opt.input_nc, opt.ndf, opt.netD,
                                                       opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type,
                                                       opt.init_gain,
                                                       self.gpu_ids)
                    self.loss_names.append('D_Seg')
                    self.loss_names.append('D_Seg2')
                    self.model_names.append('D_Seg')
                    self.model_names.append('D_Seg2')

                    self.optimizer_D_Seg = torch.optim.Adam(itertools.chain(self.netD_Seg.parameters(),
                                                                            self.netD_Seg2.parameters()),
                                                            lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_D_Seg)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

        if self.segmentation:
            self.fake_B_seg_binary, self.fake_B_seg_multi_class = self.segModel.segment_images(self.fake_B)
            self.real_B_seg_binary, self.real_B_seg_multi_class = self.segModel.segment_images(self.real_B)

    def backward_D_basic(self, netD, real_AB, fake_AB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        loss_D.backward()

        return loss_D

    def backward_D_Base(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        real_AB = torch.cat((self.real_A, self.real_B), 1)

        self.loss_D = self.backward_D_basic(self.netD, real_AB, fake_AB)

    def backward_D_Seg_binary(self):

        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_B, self.fake_B_seg_binary), 1))
        real_AB = torch.cat((self.real_B, self.real_B_seg_binary), 1)

        self.loss_D_Seg = self.backward_D_basic(self.netD_Seg, real_AB, fake_AB)

    def backward_D_Seg_multi_class(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_B, self.fake_B_seg_multi_class), 1))
        real_AB = torch.cat((self.real_B, self.real_B_seg_multi_class), 1)

        self.loss_D_Seg2 = self.backward_D_basic(self.netD_Seg2, real_AB, fake_AB)

    def backward_G(self):
        lambda_Seg = self.lambda_Seg

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.loss_G_GAN = self.criterionGAN(self.netD(fake_AB), True)

        if self.segmentation:
            if self.opt.segmentation_output == "binary":
                fake_AB = torch.cat((self.real_B, self.fake_B_seg_binary), 1)
                self.loss_G_Seg = self.criterionGAN(self.netD_Seg(fake_AB), True)
            elif self.opt.segmentation_output == "multi_class":
                fake_AB = torch.cat((self.real_B, self.fake_B_seg_multi_class), 1)
                self.loss_G_Seg2 = self.criterionGAN(self.netD_Seg2(fake_AB), True)
            else:
                fake_AB1 = torch.cat((self.real_B, self.fake_B_seg_binary), 1)
                fake_AB2 = torch.cat((self.real_B, self.fake_B_seg_multi_class), 1)
                self.loss_G_Seg = self.criterionGAN(self.netD_Seg(fake_AB1), True)
                self.loss_G_Seg2 = self.criterionGAN(self.netD_Seg2(fake_AB2), True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        if self.segmentation:
            if self.opt.segmentation_output == "binary":
                self.loss_G += lambda_Seg * self.loss_G_Seg
            elif self.opt.segmentation_output == "multi_class":
                self.loss_G += lambda_Seg * self.loss_G_Seg2
            elif self.opt.segmentation_output == "both":
                self.loss_G += lambda_Seg * self.loss_G_Seg + lambda_Seg * self.loss_G_Seg2

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        if self.segmentation:
            if self.opt.segmentation_output == "binary":
                self.set_requires_grad([self.netD, self.netD_Seg], True)
            elif self.opt.segmentation_output == "multi_class":
                self.set_requires_grad([self.netD, self.netD_Seg2], True)
            elif self.opt.segmentation_output == "both":
                self.set_requires_grad([self.netD, self.netD_Seg, self.netD_Seg2], True)
        else:
            self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        if self.segmentation:
            self.optimizer_D_Seg.zero_grad()
        self.backward_D_Base()

        if self.segmentation:
            if self.opt.segmentation_output == "binary":
                self.backward_D_Seg_binary()
            elif self.opt.segmentation_output == "multi_class":
                self.backward_D_Seg_multi_class()
            else:
                self.backward_D_Seg_binary()
                self.backward_D_Seg_multi_class()

        self.optimizer_D.step()
        if self.segmentation:
            self.optimizer_D_Seg.step()

        # update G
        if self.segmentation:
            if self.opt.segmentation_output == "binary":
                self.set_requires_grad([self.netD, self.netD_Seg], False)
            elif self.opt.segmentation_output == "multi_class":
                self.set_requires_grad([self.netD, self.netD_Seg2], False)
            elif self.opt.segmentation_output == "both":
                self.set_requires_grad([self.netD, self.netD_Seg, self.netD_Seg2], False)
        else:
            self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
