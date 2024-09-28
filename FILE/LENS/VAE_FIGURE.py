import os
import time
import json
import h5py
import numpy
import corner
import argparse
from matplotlib import pyplot, colors

def plot_loss(epoch_size, history):
    
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
    
    epoch = numpy.arange(1, epoch_size + 1)
    
    plot.plot(epoch, history['all_loss'], color = 'black', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{all}$')
    
    plot.plot(epoch, history['mean_loss'], color = 'purple', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{mean}$')
    
    plot.plot(epoch, history['scatter_loss'], color = 'blue', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{scatter}$')
    
    plot.plot(epoch, history['covariance_loss'], color = 'green', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{covariance}$')
    
    plot.plot(epoch, history['divergence_loss'], color = 'orange', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{divergence}$')
    
    plot.plot(epoch, history['val_all_loss'], color = 'black', linestyle = '-', linewidth = 2.5)
    
    plot.plot(epoch, history['val_mean_loss'], color = 'purple', linestyle = '-', linewidth = 2.5)
    
    plot.plot(epoch, history['val_scatter_loss'], color = 'blue', linestyle = '-', linewidth = 2.5)
    
    plot.plot(epoch, history['val_covariance_loss'], color = 'green', linestyle = '-', linewidth = 2.5)
    
    plot.plot(epoch, history['val_divergence_loss'], color = 'orange', linestyle = '-', linewidth = 2.5)
    
    plot.set_yscale('log')
    plot.set_xscale('linear')
    
    plot.legend()
    plot.set_xlim(1, epoch_size)
    
    plot.set_xlabel(r'$\mathrm{Epochs}$')
    plot.set_ylabel(r'$\mathrm{Losses}$')
    
    return figure


def plot_data(phi, bin_size, grid_size):
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_delta = 0.01
    z_data = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, grid_size)
    
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
    color_list = ['darkmagenta', 'darkblue', 'darkcyan', 'darkgreen', 'darkred']
    
    for m in range(bin_size):
        
        phi0 = numpy.mean(phi[:, m, :], axis = 0)
        
        plot.plot(z_data, phi0, color = color_list[m], linewidth = 2.0, label = r'$m = {:.0f}$'.format(m + 1))
        
        plot.plot(z_data, numpy.transpose(phi[:, m, :]), color = color_list[m], linewidth = 0.005, alpha = 0.005)
    
    plot.legend()
    plot.set_xlim(z1, z2)
    plot.set_ylim(0.0, 8.0)
    
    plot.set_xlabel(r'$z$')
    plot.set_ylabel(r'$\phi^m (z)$')
    
    return figure

def plot_covariance(phi, bin_size, grid_size):
    """
    Plot the covariance.
    
    Arguments:
        phi (numpy.ndarray): The data.
        bin_size (int): The bin size.
        grid_size (int): The grid size.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    """
    data = phi.reshape(-1, bin_size * grid_size)
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    covariance = numpy.cov(data, rowvar = False)
    norm = colors.Normalize(vmin = -0.1, vmax = +0.1)
    
    for m in range(bin_size):
        for n in range(bin_size):
            
            map = covariance[n * grid_size: n * grid_size + grid_size // 2, m * grid_size: m * grid_size + grid_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    
    return figure

def plot_latent(latent_data, latent_size):
    
    figure = pyplot.figure(figsize = (3 * latent_size, 3 * latent_size))
    
    figure = corner.corner(
        latent_data, 
        figure=figure,
        show_titles=True, 
        quantiles=[0.16, 0.5, 0.84], 
        range=[(-3.0, 3.0)] * latent_size,
        labels=[r'$\xi_{}$'.format(i + 1) for i in range(latent_size)]
    )
    
    return figure


def main(path, name):
    '''
    Main function for the PCA analysis.
    
    Arguments:
        path (str): The path to the base folder.
        name (str): The name of the sample datasets.
    
    Returns:
        duration (float): The duration of the PCA analysis
    '''
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    os.makedirs(os.path.join(plot_path, '{}/VAE/'.format(name)), exist_ok = True)
    
    # Meta
    with open(os.path.join(data_path, '{}/INFO.json'.format(name)), 'r') as file:
        info = json.load(file)
    
    history = info['HISTORY']
    bin_size = info['META']['BIN_SIZE']
    grid_size = info['META']['GRID_SIZE']
    mock_size = info['META']['MOCK_SIZE']
    epoch_size = info['META']['EPOCH_SIZE']
    latent_size = info['META']['LATENT_SIZE']
    
    # Model
    with h5py.File(os.path.join(data_path, '{}/MODEL.hdf5'.format(name)), 'r') as file:
        test_input = numpy.array(file['TEST_INPUT'])
        test_output = numpy.array(file['TEST_OUTPUT'])
        latent_data = numpy.array(file['LATENT_DATA'])
    
    # Figure
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Plot
    plot_size = mock_size // 40
    select = numpy.random.choice(numpy.arange(mock_size), plot_size, replace = False)
    
    # Loss
    figure = plot_loss(epoch_size, history)
    figure.savefig(os.path.join(plot_path, '{}/VAE/LOSS.pdf'.format(name)), bbox_inches = 'tight')
    figure.clear()
    
    # Data
    figure = plot_data(test_input[select, :, :], bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/INPUT_DATA.pdf'.format(name)), bbox_inches = 'tight')
    figure.clear()
    
    figure = plot_data(test_output[select, :, :], bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/OUTPUT_DATA.pdf'.format(name)), bbox_inches = 'tight')
    figure.clear()
    
    # Covariance
    figure = plot_covariance(test_input, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/INPUT_COVARIANCE.pdf'.format(name)), bbox_inches = 'tight')
    figure.clear()
    
    figure = plot_covariance(test_output, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/OUTPUT_COVARIANCE.pdf'.format(name)), bbox_inches = 'tight')
    figure.clear()
    
    # Latent
    figure = plot_latent(latent_data, latent_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/LATENT.pdf'.format(name)), bbox_inches = 'tight')
    figure.clear()
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB VAE')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the sample datasets')
    
    PATH = PARSE.parse_args().path
    NAME = PARSE.parse_args().name
    RESULT = main(PATH, NAME)