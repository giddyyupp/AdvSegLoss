from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--cityscapes', type=bool, default=False, help='test with cityscapes dataset')
        parser.add_argument('--cityscape_fnames',
                            type=str, default="./datasets/cityscapes_test_file_names.txt",
                            help='correct cityscape file names')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', default=False, help='use eval mode during test time.')  # action='store_true',
        parser.add_argument('--num_test', type=int, default=500000, help='how many test images to run')  # for style 751, for content 500

        parser.set_defaults(model='test')  # TODO: change to test if using cyclegan, pix2pix for Pix2Pix!
        # parser.set_defaults(model='pix2pix')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
