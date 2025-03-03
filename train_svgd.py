"""
Stein Variational Gradient Descent for Deep ConvNet on GPU.
Current implementation is mainly using for-loops over model instances.
The codes are modified after Dr. Yinhao Zhu's repository: https://github.com/cics-nd/cnn-surrogate
"""

import torch
import numpy as np
from time import time
from args import args
from models.RRDB import Net
from models.bayes_nn import BayesNN
from models.svgd import SVGD
from utils.funcs import load_data, plot_pred, plot_nse, save_stats
from utils.misc import mkdirs, logger
import json
import sys
import scipy.io


# train the network on GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])

# # load data
train_loader, test_loader, xtest, yhtest, ytest, stats, land_mask = load_data(args)
logger['train_output_var'] = stats['ytrain_var']
logger['test_output_var'] = stats['ytest_var']
args.ntrain = len(train_loader.dataset)
print('Number of training samples: {}'.format(args.ntrain))
print('Loaded data!')

# deterministic NN
model = Net(stats['ic'], stats['oc'], nf=args.features,act_fun=args.act_fun).to(device)

print(model)
# Bayesian NN
bayes_nn = BayesNN(model, n_samples=args.n_samples).to(device)

# Initialize SVGD
svgd = SVGD(bayes_nn, train_loader)


def test(epoch, logger):
    """Evaluate model during training. 
    Print predictions including 4 rows:
        1. target
        2. predictive mean
        3. error of the above two
        4. two sigma of predictive variance
    """
    bayes_nn.eval()
    
    mse_test, nlp_test = 0., 0.
    for batch_idx, (input, hist, target) in enumerate(test_loader):
        input, hist, target = input.to(device), hist.to(device), target.to(device)
        mse, nlp, output = bayes_nn._compute_mse_nlp(input, hist, target,
                            size_average=True, out=True)
        # output: S x N x oC x oH x oW --> N x oC x oH x oW
        y_pred_mean = output.mean(0)
        EyyT = (output ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        y_noise_var = (- bayes_nn.log_beta).exp().mean()
        y_pred_var =  EyyT - EyEyT + y_noise_var

        mse_test += mse.item()
        nlp_test += nlp.item()
    rmse_test = np.sqrt(mse_test / len(test_loader))
    r2_test = 1 - mse_test * target.numel() / logger['test_output_var']
    mnlp_test = nlp_test / len(test_loader)
    logger['rmse_test'].append(rmse_test)
    logger['r2_test'].append(r2_test)
    logger['mnlp_test'].append(mnlp_test)
    print("epoch {}, testing  r2: {:.4f}, test mnlp: {:.2f}".format(
            epoch, r2_test, mnlp_test))

    if epoch % args.plot_freq == 0:
        # plot predictions
        n_samples = 3
        idx = torch.LongTensor(np.random.choice(xtest.shape[0] - args.ntf + 1, n_samples, replace=False))
        for i in range(n_samples):
            x = xtest[[idx[i]]]
            yh = yhtest[[idx[i]]]
            ytarget = ytest[idx[i]]
            x = (torch.FloatTensor(x)).to(device)
            yh = (torch.FloatTensor(yh)).to(device)
            ypred, ypred_var = bayes_nn.predict(x,yh)
            ypred, ypred_var = ypred.data.cpu().numpy(), ypred_var.data.cpu().numpy()
            data = np.concatenate((ytarget, ypred[0], ytarget - ypred[0], np.sqrt(ypred_var[0])), axis=0)
            plot_pred(data, epoch, idx[i], land_mask, args, args.pred_dir)
    if epoch == args.epochs:
        # save the final predictions
        n_samples = ytest.shape[0]
        ypred, ypred_var = np.zeros_like(ytest), np.zeros_like(ytest)
        for i in range(n_samples):
            x = xtest[[i]]
            yh = yhtest[[i]]
            x = (torch.FloatTensor(x)).to(device)
            yh = (torch.FloatTensor(yh)).to(device)
            ypredi, ypred_vari = bayes_nn.predict(x,yh)
            ypred[i], ypred_var[i] = ypredi.data.cpu().numpy(), ypred_vari.data.cpu().numpy()
        # Save the data to a .mat file
        scipy.io.savemat(args.pred_dir + '/y_ypred.mat', dict(ytarget=ytest, ypred=ypred, ypred_var=ypred_var))
        plot_nse(ytest, ypred, land_mask, args.pred_dir)


if args.pre_trained:
    # post-processing using the pretrained network ## #
    bayes_nn.load_state_dict(torch.load(args.ckpt_dir + '/model_epoch{}.pth'.format(args.epochs)))
    test(args.epochs, logger)
    sys.exit(0)


print('Start training.........................................................')
tic = time()
for epoch in range(1, args.epochs + 1):
    svgd.train(epoch, logger)
    with torch.no_grad():
        test(epoch, logger)
training_time = time() - tic
print('Finished training:\n{} epochs\n{} data\n{} samples (SVGD)\n{} seconds'
    .format(args.epochs, args.ntrain, args.n_samples, training_time))

# save training results
x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
# plot the rmse, r2-score curve and save them in txt
save_stats(args.train_dir, logger, x_axis)

args.training_time = training_time
args.n_params, args.n_layers = model._num_parameters_convlayers()
with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)
