import os
import time
import json
import h5py
import numpy
import pickle
import corner
import argparse
from tensorflow import keras
from matplotlib import cm, colors, pyplot, ticker


def plot_loss(epoch_size, history):
    
    figure, plot = pyplot.subplots(nrows = 1, ncols = 2, figsize = (16, 8))
    
    epoch = numpy.arange(1, epoch_size + 1)
    
    plot[0].plot(epoch, history['all_loss'], color = 'black', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{all}$')
    
    plot[0].plot(epoch, history['mean_loss'], color = 'purple', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{mean}$')
    
    plot[0].plot(epoch, history['scatter_loss'], color = 'blue', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{scatter}$')
    
    plot[0].plot(epoch, history['covariance_loss'], color = 'green', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{covariance}$')
    
    plot[0].plot(epoch, history['divergence_loss'], color = 'orange', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{divergence}$')
    
    plot[0].set_yscale('log')
    plot[0].set_xscale('linear')
    
    plot[0].set_xlim(1, epoch_size)
    plot[0].set_ylim(1.0e-10, 1.0e+2)
    
    plot[0].set_title(r'$\mathrm{Train}$')
    plot[0].set_xlabel(r'$\mathrm{Epochs}$')
    plot[0].set_ylabel(r'$\mathrm{Losses}$')
    
    plot[1].plot(epoch, history['val_all_loss'], color = 'black', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{all}$')
    
    plot[1].plot(epoch, history['val_mean_loss'], color = 'purple', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{mean}$')
    
    plot[1].plot(epoch, history['val_scatter_loss'], color = 'blue', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{scatter}$')
    
    plot[1].plot(epoch, history['val_covariance_loss'], color = 'green', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{covariance}$')
    
    plot[1].plot(epoch, history['val_divergence_loss'], color = 'orange', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{divergence}$')
    
    plot[1].set_yscale('log')
    plot[1].set_xscale('linear')
    
    plot[1].set_xlim(1, epoch_size)
    plot[1].set_ylim(1.0e-10, 1.0e+2)
    
    plot[1].set_yticklabels([])
    plot[1].legend(loc='upper right')
    
    plot[1].set_xlabel(r'$\mathrm{Epochs}$')
    plot[1].set_title(r'$\mathrm{Validation}$')
    
    figure.subplots_adjust(wspace = 0.00)
    return figure


def plot_data(phi, bin_size, data_size):
    '''
    Plot the data.
    
    Arguments:
        phi (numpy.ndarray): The data.
        z_grid (numpy.ndarray): The grid.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    '''
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, data_size)
    
    # Plot
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
    color_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    
    for m in range(bin_size):
        
        plot.plot(z_grid, numpy.transpose(phi[:, m, :]), color = color_list[m].replace('dark', ''), linewidth = 0.05, alpha = 0.05)
        
        plot.plot(z_grid, numpy.mean(phi[:, m, :], axis = 0), color = color_list[m], linewidth = 5.0, label = r'$m = {:.0f}$'.format(m + 1))
    
    plot.legend()
    plot.set_xlim(z1, z2)
    plot.set_ylim(0.0, 8.0)
    
    plot.set_xlabel(r'$z$')
    plot.set_ylabel(r'$\phi^m (z)$')
    
    return figure


def plot_covariance(phi, bin_size, data_size):
    '''
    Plot the covariance.
    
    Arguments:
        phi (numpy.ndarray): The data.
        bin_size (int): The bin size.
        data_size (int): The grid size.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    '''
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    covariance = numpy.cov(numpy.reshape(phi, (-1, bin_size * data_size)), rowvar = False)
    norm = colors.Normalize(vmin = -0.25, vmax = +0.25)
    
    for m in range(bin_size):
        for n in range(bin_size):
            
            map = covariance[n * data_size: n * data_size + data_size // 2, m * data_size: m * data_size + data_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    
    return figure


def plot_delta(input_phi, output_phi, bin_size, data_size):
    """
    Plot the covariance.
    
    Arguments:
        input_phi (numpy.ndarray): The input data.
        output_phi (numpy.ndarray): The output data.
        bin_size (int): The size of tomographic bins.
        data_size (int): The size of redshift grid points.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    """
    input_data = input_phi.reshape((-1, bin_size * data_size))
    output_data = output_phi.reshape((-1, bin_size * data_size))
    
    input_covariance = numpy.cov(input_data, rowvar = False)
    output_covariance = numpy.cov(output_data, rowvar = False)
    covariance = numpy.subtract(input_covariance, output_covariance)
    
    norm = colors.Normalize(vmin = -0.25, vmax = +0.25)
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    for m in range(bin_size):
        for n in range(bin_size):
            
            map = covariance[n * data_size: n * data_size + data_size // 2, m * data_size: m * data_size + data_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\Delta \mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    
    return figure

def plot_latent(latent_data, latent_size):
    
    figure = pyplot.figure(figsize = (3 * latent_size, 3 * latent_size))
    
    figure = corner.corner(
        latent_data, 
        figure=figure,
        title_fmt='.1f',
        show_titles=True, 
        quantiles=[0.16, 0.50, 0.84], 
        range=[(-3.0, 3.0)] * latent_size, 
        labels=[r'$\xi_{}$'.format(i + 1) for i in range(latent_size)]
    )
    
    return figure

def plot_variable(pca_model, vae_model, bin_size, data_size, latent_size):
    
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, data_size)
    
    latent1 = -5.0
    latent2 = +5.0
    sample_size = 10
    latent_point = numpy.zeros(sample_size + 1)
    latent_sample = numpy.linspace(latent1, latent2, sample_size + 1)
    
    figure, plot = pyplot.subplots(nrows = latent_size, ncols = 1, figsize = (15, 5 * latent_size))
    
    for i in range(latent_size):
        
        latent_data = numpy.zeros((sample_size + 1, latent_size))
        for j in range(latent_size):
            if j == i:
                latent_data[:, j] = latent_sample
            else:
                latent_data[:, j] = latent_point
        phi_data = numpy.reshape(pca_model.inverse_transform((vae_model.decoder(latent_data))), (-1, bin_size, data_size))
        
        norm = colors.Normalize(vmin=latent1, vmax=latent2)
        scalar_mappable = cm.ScalarMappable(cmap=cm.plasma_r, norm=norm)
        color_list = cm.plasma_r(numpy.linspace(0.0, 1.0, sample_size + 1))
        
        for n in range(sample_size + 1):
            for m in range(bin_size):
                plot[i].plot(z_grid, phi_data[n, m, :], color = color_list[n], linewidth = 1.0)
        
        plot[i].set_xlim(z1, z2)
        plot[i].set_ylim(0.0, 8.0)
        
        plot[i].set_xlabel(r'$z$')
        plot[i].set_ylabel(r'$\phi^m (z)$')
        
        plot[i].yaxis.set_major_locator(ticker.FixedLocator(numpy.arange(0.0, 8.0, 1.0)))
        if i < latent_size - 1:
            plot[i].set_xticklabels([])
        
        figure.colorbar(mappable=scalar_mappable, ax=plot[i], orientation='vertical', label=r'$\xi_{}$'.format(i + 1), shrink=0.8)
    figure.subplots_adjust(wspace = 0.05, hspace = 0.00)
    return figure


def main(name, folder):
    '''
    Plot the VAE figure.
    
    Arguments:
        name (str): The name of the datasets.
        folder (str): The base folder of the datasets.
    
    Returns:
        duration (float): The duration of the PCA analysis
    '''
    # Start
    start = time.time()
    
    # Path
    path = os.path.join(folder, '{}/'.format(name))
    
    # Meta
    with open(os.path.join(path, 'VAE/INFO.json'), 'r') as file:
        info = json.load(file)
    
    history = info['HISTORY']
    bin_size = info['META']['BIN_SIZE']
    data_size = info['META']['DATA_SIZE']
    mock_size = info['META']['MOCK_SIZE']
    epoch_size = info['META']['EPOCH_SIZE']
    latent_size = info['META']['LATENT_SIZE']
    
    # Data
    with h5py.File(os.path.join(path, 'VAE/DATA.hdf5'), 'r') as file:
        test_input = numpy.array(file['TEST_INPUT'])
        test_output = numpy.array(file['TEST_OUTPUT'])
        latent_data = numpy.array(file['LATENT_DATA'])
    
    # Model
    vae_model = keras.models.load_model(os.path.join(path, 'VAE/MODEL'))
    with open(os.path.join(path, 'PCA/MODEL.pkl'), 'rb') as file:
        pca_model = pickle.load(file)
    
    # Figure
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    test_size = mock_size // 10
    plot_size = mock_size // 50
    select = numpy.random.choice(numpy.arange(test_size), plot_size, replace = False)
    
    # Loss
    figure = plot_loss(epoch_size, history)
    figure.savefig(os.path.join(path, 'VAE/LOSS.png'), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Data
    figure = plot_data(test_output[select, :, :], bin_size, data_size)
    figure.savefig(os.path.join(path, 'VAE/OUTPUT_DATA.png'), bbox_inches = 'tight', dpi = 512)
    pyplot.close(figure)
    
    # Covariance
    figure = plot_covariance(test_output, bin_size, data_size)
    figure.savefig(os.path.join(path, 'VAE/OUTPUT_COVARIANCE.png'), bbox_inches = 'tight', dpi = 512)
    pyplot.close(figure)
    
    # Delta
    figure = plot_delta(test_input, test_output, bin_size, data_size)
    figure.savefig(os.path.join(path, 'VAE/DELTA.png'), bbox_inches = 'tight', dpi = 512)
    pyplot.close(figure)
    
    # Latent
    figure = plot_latent(latent_data, latent_size)
    figure.savefig(os.path.join(path, 'VAE/LATENT.png'), bbox_inches = 'tight', dpi = 512)
    pyplot.close(figure)
    
    # Variable
    figure = plot_variable(pca_model, vae_model, bin_size, data_size, latent_size)
    figure.savefig(os.path.join(path, 'VAE/VARIABLE.png'), bbox_inches = 'tight', dpi = 512)
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='VAE Figure')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the datasets.')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets.')
    
    # Parse
    NAME = PARSE.parse_args().name
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NAME, FOLDER)