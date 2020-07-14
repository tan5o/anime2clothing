from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument("--dataset_root", default="dataset/")
        self.parser.add_argument('--dataset', required=True, help='facades')
        self.parser.add_argument('--start_resolution', type=int, default=8, help='the starting resolution')
        self.parser.add_argument("--gan_mode", default="lsgan")
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--start_epoch', type=int, default=0, help='the starting epoch')
        self.parser.add_argument('--load_epoch', type=int, default=0, help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=70, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=30, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--threads', type=int, default=12, help='number of threads for data loader to use')
        self.parser.add_argument('--l1_loss', action='store_true', help='include or not l1 loss?')
        self.parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
        self.parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        self.parser.add_argument('--no_ic_loss', action='store_true')
        self.parser.add_argument('--batch_rate', type=int, default=1, help='rate of batch size')

        # fot generator
        self.parser.add_argument("--g_norm", default="batch", help='normalization type of geneartor')
        self.parser.add_argument("--g_conv", default="normal", help='convolution type of generator')

        # for discriminators
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument("--d_norm", default="none", help='normalization type of discriminator')
        self.parser.add_argument("--d_conv", default="normal", help='convolution type of discriminator')

        self.isTrain = True



MINIBATCH_MAPPING = {
    4: 128,
    8: 64,
    16: 128,
    32: 64,
    64: 32,
    128: 16,
    256: 4
}

EPOCH_MAPPING = {
    4: 0,
    8: 7,
    16: 15,
    32: 15,
    64: 30,
    128: 60,
    256: 100
}
