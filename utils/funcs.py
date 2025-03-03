import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import h5py
from datetime import datetime
import torch as th
plt.switch_backend('agg')


def load_data(args):
    with h5py.File(args.data_dir + "in_output_data_{}_{}_ERA5.hdf5".format(args.res,args.dataset), 'r') as f:
        x = f['x'][()]  # Nx*C*H*W, auxillary data (P, T, rTWSA; C=3)
        y = f['y'][()]  # Nx*H*W, GRACE/FO TWSA data
        land_mask = f['land_mask'][()]

    # lat: -60 to 84, lon: -180 to 180, H: 144, W: 360
    x, y, land_mask = x[:,:,:144], y[:,:144], land_mask[:144]
    x[x!=x], y[y!=y] = 0., 0.  # set the NaN elements as 0.
    y = np.expand_dims(y, axis=1)

    # remove the auxiliary between 2000.01 and 2002.03
    year0 = 2000
    start_date = datetime(year0, 1, 1)
    end_date = datetime(2002, 4, 1)
    id_month0 = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
    x = x[id_month0:id_month0 + y.shape[0]]
    # use the first 183 months (i.e., the GRACE period) for training,
    # the data gaps within GRACE period have been filled by interpolation
    xtrain, ytrain = x[:183], y[:183]
    # use the GRACE-FO data for testing, the 11-month data gaps between GRACE and GRACE-FO have been filled by nans
    xtest, ytest = x[194:], y[194:]
    # reorganize the data, x: N0*C*H*W -> N*nt*C*H*W, y: N0*1*H*W -> N*ntf*H*W
    xtrain, yhtrain, ytrain = data_org(xtrain, ytrain, args)
    xtest, yhtest, ytest = data_org(xtest, ytest, args)
    print('Shape of xtrain: {},yhtrain:{}, ytrain: {}, xtest: {}, yhtest: {}, ytest: {}'
          .format(xtrain.shape,yhtrain.shape, ytrain.shape,xtest.shape,yhtest.shape,ytest.shape))

    # calculate the statistics of the data
    ytrain_mean = np.mean(ytrain, 0)
    ytrain_var = np.sum((ytrain - ytrain_mean) ** 2)
    stats = {}
    stats['ytrain_mean'] = ytrain_mean
    stats['ytrain_var'] = ytrain_var

    ytest_mean = np.mean(ytest, 0)
    ytest_var = np.sum((ytest - ytest_mean) ** 2)
    stats['ytest_mean'] = ytest_mean
    stats['ytest_var'] = ytest_var
    stats['ic'], stats['oc'] = xtest.shape[1]*xtest.shape[2]+yhtest.shape[1]*yhtest.shape[2], ytest.shape[1]

    # create data loaders
    data_train = th.utils.data.TensorDataset(th.FloatTensor(xtrain),
                                             th.FloatTensor(yhtrain),
                                             th.FloatTensor(ytrain))
    data_test = th.utils.data.TensorDataset(th.FloatTensor(xtest),
                                            th.FloatTensor(yhtest),
                                            th.FloatTensor(ytest))
    train_loader = th.utils.data.DataLoader(data_train,
                                            batch_size=args.batch_size,
                                            shuffle=True)
    test_loader = th.utils.data.DataLoader(data_test,
                                           batch_size=args.batch_size,
                                           shuffle=True)
    npixel_train = len(train_loader.dataset) * train_loader.dataset[0][1].numel()
    npixel_test = len(test_loader.dataset) * test_loader.dataset[0][1].numel()
    stats['npixel_train'] = npixel_train
    stats['npixel_test'] = npixel_test

    return train_loader, test_loader, xtest, yhtest, ytest, stats, land_mask


def data_org(x, y, args, stride=1):
    n0, c, h, w = x.shape
    input_window = args.nt + args.ntf
    n = (n0 - input_window) // stride + 1

    X = np.zeros([input_window, n, c, h, w])
    Y = np.zeros([input_window, n, 1, h, w])
    for i in range(n):
        start_x = stride * i
        end_x = start_x + input_window
        X[:, i] = x[start_x:end_x]
        Y[:, i] = y[start_x:end_x]
    # T*N*C*H*W -> N*T*C*H*W
    X, Y = np.transpose(X, (1, 0, 2, 3, 4)), np.transpose(Y, (1, 0, 2, 3, 4))
    xnew, yhist = X, Y[:,:args.nt]
    ynew = Y[:, args.nt:,0]

    return xnew, yhist, ynew


def plot_pred(data, epoch, idx, land_mask, args, output_dir):
    land_mask = np.expand_dims(land_mask,axis=0)
    land_mask = np.repeat(land_mask, data.shape[0], axis=0)
    data[land_mask!=land_mask] = np.nan
    cols, rows = args.ntf, 4
    vmaxs = np.full((cols*rows,1), np.nan)
    vmaxs[:cols*(rows-2)] = 30
    vmaxs[cols * (rows - 2):cols * (rows - 1)] = 10
    vmaxs[cols * (rows - 1):] = 5
    vmins = -vmaxs
    vmins[cols * (rows - 1):] = 0
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4,rows*2))
    plt.subplots_adjust(hspace=0.16, wspace=0.15)
    ylabels = (['Observations','Predictions','Residuals','STDs'])
    fs = 13 # font size
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        cax = ax.imshow(data[j], cmap='jet', origin='lower', vmin=vmins[j], vmax=vmaxs[j])
        if j < cols * (rows - 1):
            cbar = plt.colorbar(cax, ax=ax, fraction=0.019, pad=0.02,extend='both')
        else:
            cbar = plt.colorbar(cax, ax=ax, fraction=0.019, pad=0.02, extend='max')
        cbar.ax.tick_params(axis='both', which='both', length=0)
        cbar.ax.yaxis.get_offset_text().set_fontsize(fs-1)
        cbar.ax.tick_params(labelsize=fs-1)
        if j < cols:
            ax.set_title('t+{:1}'.format(j+1), fontsize=fs)
        if j%cols==0:
            ax.set_ylabel(ylabels[j//cols], fontsize=fs)

    print("epoch {}, done with printing sample output {}".format(epoch, idx))
    plt.savefig(output_dir+'/epoch{}_output{}.png'.format(epoch, idx), bbox_inches='tight',dpi=500)
    plt.close(fig)
    

def plot_nse(observed, predicted, land_mask, output_dir):
    _, nt, h, w = observed.shape
    NSE = np.full((nt,h,w),np.nan)
    for i in range(nt):
        for j in range(h):
            for k  in range(w):
                if not np.isnan(land_mask[j,k]):
                    NSE[i,j,k] = nse(observed[:,i,j,k], predicted[:,i,j,k])

    cols, rows = nt, 1
    vmin, vmax = 0, 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 4))
    plt.subplots_adjust(hspace=0.16, wspace=0.15)
    fs = 13  # font size
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        cax = ax.imshow(NSE[j], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.019, pad=0.02, extend='min')
        cbar.ax.tick_params(axis='both', which='both', length=0)
        cbar.ax.yaxis.get_offset_text().set_fontsize(fs - 1)
        cbar.ax.tick_params(labelsize=fs - 1)
        ax.set_title('t+{:1}'.format(j + 1), fontsize=fs)

    print("done with plotting NSE images")
    plt.savefig(output_dir + '/NSE.png', bbox_inches='tight', dpi=500)
    plt.close(fig)


def nse(observed, predicted):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) coefficient.

    Parameters:
    observed (array-like): Observed data.
    predicted (array-like): predicted data.

    Returns:
    float: NSE value.
    """
    observed = np.array(observed)
    predicted = np.array(predicted)
    mean_observed = np.mean(observed)

    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)

    nse_value = 1 - (numerator / denominator)
    return nse_value


def save_stats(save_dir, logger, x_axis):
    rmse_train = logger['rmse_train']
    rmse_test = logger['rmse_test']
    r2_train = logger['r2_train']
    r2_test = logger['r2_test']

    if 'mnlp_test' in logger.keys():
        mnlp_test = logger['mnlp_test']
        if len(mnlp_test) > 0:
            plt.figure()
            plt.plot(x_axis, mnlp_test, label="Test: {:.3f}".format(np.mean(mnlp_test[-5:])))
            plt.xlabel('Epoch')
            plt.ylabel('MNLP')
            plt.legend(loc='upper right')
            plt.savefig(save_dir + "/mnlp_test.pdf", dpi=600)
            plt.close()
            np.savetxt(save_dir + "/mnlp_test.txt", mnlp_test)

    if 'log_beta' in logger.keys():
        log_beta = logger['log_beta']
        if len(log_beta) > 0:
            plt.figure()
            plt.plot(x_axis, log_beta, label="Test: {:.3f}".format(np.mean(log_beta[-5:])))
            plt.xlabel('Epoch')
            plt.ylabel('Log-Beta (noise precision)')
            plt.legend(loc='upper right')
            plt.savefig(save_dir + "/log_beta.pdf", dpi=600)
            plt.close()
            np.savetxt(save_dir + "/log_beta.txt", log_beta)

    plt.figure()
    plt.plot(x_axis, r2_train, label="Train: {:.3f}".format(np.mean(r2_train[-5:])))
    plt.plot(x_axis, r2_test, label="Test: {:.3f}".format(np.mean(r2_test[-5:])))
    plt.xlabel('Epoch')
    plt.ylabel(r'$R^2$-score')
    plt.legend(loc='lower right')
    plt.savefig(save_dir + "/r2.pdf", dpi=600)
    plt.close()
    np.savetxt(save_dir + "/r2_train.txt", r2_train)
    np.savetxt(save_dir + "/r2_test.txt", r2_test)

    plt.figure()
    plt.plot(x_axis, rmse_train, label="train: {:.3f}".format(np.mean(rmse_train[-5:])))
    plt.plot(x_axis, rmse_test, label="test: {:.3f}".format(np.mean(rmse_test[-5:])))
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    plt.savefig(save_dir + "/rmse.pdf", dpi=600)
    plt.close()
    np.savetxt(save_dir + "/rmse_train.txt", rmse_train)
    np.savetxt(save_dir + "/rmse_test.txt", rmse_test)