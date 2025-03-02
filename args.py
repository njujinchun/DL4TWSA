import argparse
import torch
import json
import random
from pprint import pprint
from utils.misc import mkdirs


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Bayesian Convolutional Encoder-Decoder Networks with SVGD')
        self.add_argument('--pre-trained', action='store_true', default=False, help='test using the pre-trained model or not.')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')

        # model
        self.add_argument('--n-samples', type=int, default=20, help='(20-30) number of model instances in SVGD')
        self.add_argument('--features', type=int, default=64, help='number of basic features in basic conv layers')
        self.add_argument('--ss', action='store_true', default=True, help='use series stationarization or not')

        # data
        self.add_argument('--data-dir', type=str, default="./datasets/", help='data directory')
        self.add_argument('--res', type=str, default='1deg', help="spatial resolution of data")
        self.add_argument('--dataset', type=str, default='JPL', help="the GRACE/FO product used for training")
        self.add_argument('--ntrain', type=int, default=0, help="number of training samples, this will be updated after loading data")
        self.add_argument('--nt', type=int, default=12, help="number of historical months considered")
        self.add_argument('--ntf', type=int, default=3, help="number of future months considered")
        
        # training
        self.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.0025, help='learnign rate')
        self.add_argument('--lr-noise', type=float, default=0.01, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=12, help='batch size for training')
        self.add_argument('--test-batch-size', type=int, default=12, help='batch size for testing')
        self.add_argument('--seed', type=int, default=None, help='manual seed used in Tensor')

        # logging
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-freq', type=int, default=50, help='how many epochs to wait before plotting test output')
        self.add_argument('--ckpt-freq', type=int, default=100, help='how many epochs to wait before saving model')
        self.add_argument('--ckpt-epoch', type=int, default=None, help='which epoch of checkpoints to be loaded in post mode')

    def parse(self):
        args = self.parse_args()
        date = '/Feb_12'
        args.run_dir = args.exp_dir + date\
            + '/nsamples{}_nt{}_ntf{}_batch{}_lr{}_noiselr{}_epochs{}_ss{}_{}'.format(
                args.n_samples, args.nt, args.ntf, args.batch_size, args.lr,
                args.lr_noise, args.epochs, args.ss, args.dataset)

        args.ckpt_dir = args.run_dir + '/checkpoints'
        mkdirs([args.run_dir, args.ckpt_dir])
        
        assert args.epochs % args.ckpt_freq == 0, 'epochs must'\
            'be dividable by ckpt_freq'

        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))

        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)

        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
