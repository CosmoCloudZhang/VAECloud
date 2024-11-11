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


def plot_data(phi, bin_size, grid_size):
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
    z_delta = (z2 - z1) / grid_size
    z_data = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, grid_size)
    
    # Plot
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
    color_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    
    for m in range(bin_size):
        
        phi0 = numpy.mean(phi[:, m, :], axis = 0)
        
        plot.plot(z_data, phi0, color = color_list[m], linewidth = 2.5, label = r'$m = {:.0f}$'.format(m + 1))
        
        plot.plot(z_data, numpy.transpose(phi[:, m, :]), color = color_list[m], linewidth = 0.025, alpha = 0.0125)
    
    plot.legend()
    plot.set_ylim(0.0, 8.0)
    plot.set_xlim(z1, z2 * 2 / 3)
    
    plot.set_xlabel(r'$z$')
    plot.set_ylabel(r'$\phi^m (z)$')
    
    return figure


def plot_covariance(phi, bin_size, grid_size):
    '''
    Plot the covariance.
    
    Arguments:
        phi (numpy.ndarray): The data.
        bin_size (int): The bin size.
        grid_size (int): The grid size.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    '''
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    covariance = numpy.cov(numpy.reshape(phi, (-1, bin_size * grid_size)), rowvar = False)
    norm = colors.Normalize(vmin = -0.05, vmax = +0.05)
    
    for m in range(bin_size):
        for n in range(bin_size):
            
            map = covariance[n * grid_size: n * grid_size + grid_size // 2, m * grid_size: m * grid_size + grid_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    
    return figure

def plot_delta(input_phi, output_phi, bin_size, grid_size):
    """
    Plot the covariance.
    
    Arguments:
        input_phi (numpy.ndarray): The input data.
        output_phi (numpy.ndarray): The output data.
        bin_size (int): The size of tomographic bins.
        grid_size (int): The size of redshift grid points.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    """
    input_data = input_phi.reshape((-1, bin_size * grid_size))
    output_data = output_phi.reshape((-1, bin_size * grid_size))
    
    input_covariance = numpy.cov(input_data, rowvar = False)
    output_covariance = numpy.cov(output_data, rowvar = False)
    covariance = numpy.subtract(input_covariance, output_covariance)
    
    norm = colors.Normalize(vmin = -0.05, vmax = +0.05)
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    for m in range(bin_size):
        for n in range(bin_size):
            
            map = covariance[n * grid_size: n * grid_size + grid_size // 2, m * grid_size: m * grid_size + grid_size // 2]
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

def plot_variable(pca_model, vae_model, bin_size, grid_size, latent_size):
    
    z1 = 0.0
    z2 = 3.0
    z_delta = (z2 - z1) / grid_size
    z_data = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, grid_size)
    
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
        phi_data = numpy.reshape(pca_model.inverse_transform((vae_model.decoder(latent_data))), (-1, bin_size, grid_size))
        
        norm = colors.Normalize(vmin=latent1, vmax=latent2)
        scalar_mappable = cm.ScalarMappable(cmap=cm.plasma_r, norm=norm)
        color_list = cm.plasma_r(numpy.linspace(0.0, 1.0, sample_size + 1))
        
        for n in range(sample_size + 1):
            for m in range(bin_size):
                plot[i].plot(z_data, phi_data[n, m, :], color = color_list[n], linewidth = 1.0)
        
        plot[i].set_ylim(0.0, 8.0)
        plot[i].set_xlim(z1, z2 * 2 / 3)
        
        plot[i].set_xlabel(r'$z$')
        plot[i].set_ylabel(r'$\phi^m (z)$')
        
        plot[i].yaxis.set_major_locator(ticker.FixedLocator(numpy.arange(0.0, 8.0, 1.0)))
        if i < latent_size - 1:
            plot[i].set_xticklabels([])
        
        figure.colorbar(mappable=scalar_mappable, ax=plot[i], orientation='vertical', label=r'$\xi_{}$'.format(i + 1), shrink=0.8)
    figure.subplots_adjust(wspace = 0.05, hspace = 0.00)
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
    
    # Data
    with h5py.File(os.path.join(data_path, '{}/DATA.hdf5'.format(name)), 'r') as file:
        test_input = numpy.array(file['TEST_INPUT'])
        test_output = numpy.array(file['TEST_OUTPUT'])
        latent_data = numpy.array(file['LATENT_DATA'])
    
    # Model
    vae_model = keras.models.load_model(os.path.join(data_path, '{}/VAE'.format(name)))
    with open(os.path.join(data_path, '{}/PCA.pkl'.format(name)), 'rb') as file:
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
    figure.savefig(os.path.join(plot_path, '{}/VAE/LOSS.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    # Data
    figure = plot_data(test_input[select, :, :], bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/INPUT_DATA.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = plot_data(test_output[select, :, :], bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/OUTPUT_DATA.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Covariance
    figure = plot_covariance(test_input, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/INPUT_COVARIANCE.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = plot_covariance(test_output, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/OUTPUT_COVARIANCE.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Delta
    figure = plot_delta(test_input, test_output, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/DELTA.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Latent
    figure = plot_latent(latent_data, latent_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/LATENT.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Variable
    figure = plot_variable(pca_model, vae_model, bin_size, grid_size, latent_size)
    figure.savefig(os.path.join(plot_path, '{}/VAE/VARIABLE.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='VAE PCA Figure')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the sample datasets')
    
    PATH = PARSE.parse_args().path
    NAME = PARSE.parse_args().name
    RESULT = main(PATH, NAME)