import argparse
import os
import pickle
import time

import h5py
import numpy
from matplotlib import cm, colors, pyplot, ticker
from tensorflow import keras


def plot_variable(z_grid, data, x1, x2, y1, y2, latent1, latent2, path):
    '''
    Draw stacked latent traversals with a shared colour bar
    
    Arguments:
        z_grid (numpy.ndarray): Redshift grid
        data (numpy.ndarray): Traversals of shape (latent, sample, bin, z)
        x1 (float): Lower redshift limit
        x2 (float): Upper redshift limit
        y1 (float): Lower amplitude limit
        y2 (float): Upper amplitude limit
        latent1 (float): Lower latent value
        latent2 (float): Upper latent value
        path (str): Output PDF path
    
    Returns:
        None
    '''
    latent_size, sample_size, bin_size, _ = data.shape
    
    figure = pyplot.figure(figsize=(12, 2.8 * latent_size))
    grid = figure.add_gridspec(
        nrows=latent_size,
        ncols=2,
        width_ratios=[1.0, 0.028],
        wspace=0.12,
        hspace=0.06,
        left=0.08,
        right=0.90,
        top=0.98,
        bottom=0.08,
    )
    
    norm = colors.Normalize(vmin=latent1, vmax=latent2)
    scalar_mappable = cm.ScalarMappable(cmap=cm.plasma_r, norm=norm)
    color_list = cm.plasma_r(numpy.linspace(0.0, 1.0, sample_size))
    
    axes = []
    for k in range(latent_size):
        ax = figure.add_subplot(grid[k, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        
        for n in range(sample_size):
            for m in range(bin_size):
                ax.plot(z_grid, data[k, n, m, :], color=color_list[n], linewidth=1.2, rasterized=True)
        
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.set_ylabel(r'$\varphi^{m}(z)$')
        ax.text(0.97, 0.88, rf'$\xi_{{{k + 1}}}$', transform=ax.transAxes, ha='right', va='top')
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        if k < latent_size - 1:
            ax.tick_params(labelbottom=False)
    
    axes[-1].set_xlabel(r'$z$')
    
    colorbar = figure.colorbar(scalar_mappable, cax=figure.add_subplot(grid[:, 1]))
    colorbar.set_label(r'$\xi_{k}$')
    colorbar.set_ticks(numpy.linspace(latent1, latent2, 5))
    colorbar.ax.tick_params(which='both', direction='in')
    colorbar.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    
    figure.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0.04, dpi=512)
    pyplot.close(figure)


def main(tag, folder):
    '''
    Plot the variable of the VAE model
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The folder containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    plot_folder = os.path.join(folder, 'PLOT/')
    lens_folder = os.path.join(folder, 'LENS/')
    source_folder = os.path.join(folder, 'SOURCE/')
    os.makedirs(os.path.join(plot_folder, f'{tag}/'), exist_ok=True)
    os.makedirs(os.path.join(plot_folder, f'{tag}/VARIABLE/'), exist_ok=True)
    
    # Lens
    vae_lens_model = keras.models.load_model(os.path.join(lens_folder, f'{tag}/VAE/'))
    with open(os.path.join(lens_folder, f'{tag}/PCA.pkl'), 'rb') as file:
        pca_lens_model = pickle.load(file)
    
    with h5py.File(os.path.join(lens_folder, f'{tag}/VAE.hdf5'), 'r') as file:
        z_size = int(file['meta']['z_size'][...])
        bin_lens_size = int(file['meta']['bin_size'][...])
        latent_lens_size = int(file['meta']['latent_size'][...])
        pca_scale_lens = float(file['meta']['pca_scale'][...])
    
    # Source
    vae_source_model = keras.models.load_model(os.path.join(source_folder, f'{tag}/VAE/'))
    with open(os.path.join(source_folder, f'{tag}/PCA.pkl'), 'rb') as file:
        pca_source_model = pickle.load(file)
    
    with h5py.File(os.path.join(source_folder, f'{tag}/VAE.hdf5'), 'r') as file:
        z_size = int(file['meta']['z_size'][...])
        bin_source_size = int(file['meta']['bin_size'][...])
        latent_source_size = int(file['meta']['latent_size'][...])
        pca_scale_source = float(file['meta']['pca_scale'][...])
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, z_size)
    
    # Latent: one coordinate at a time, others held at zero
    latent1 = -5.0
    latent2 = +5.0
    sample_size = 10
    latent_sample = numpy.linspace(latent1, latent2, sample_size + 1)
    
    data_lens = numpy.zeros((latent_lens_size, sample_size + 1, bin_lens_size, z_size))
    for k in range(latent_lens_size):
        latent_data_lens = numpy.zeros((sample_size + 1, latent_lens_size))
        latent_data_lens[:, k] = latent_sample
        coefficient_lens = numpy.array(vae_lens_model.decoder(latent_data_lens, training=False)) * pca_scale_lens
        data_lens[k] = numpy.reshape(pca_lens_model.inverse_transform(coefficient_lens), (-1, bin_lens_size, z_size))
    
    data_source = numpy.zeros((latent_source_size, sample_size + 1, bin_source_size, z_size))
    for k in range(latent_source_size):
        latent_data_source = numpy.zeros((sample_size + 1, latent_source_size))
        latent_data_source[:, k] = latent_sample
        coefficient_source = numpy.array(vae_source_model.decoder(latent_data_source, training=False)) * pca_scale_source
        data_source[k] = numpy.reshape(pca_source_model.inverse_transform(coefficient_source), (-1, bin_source_size, z_size))
    
    # Figure
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    pyplot.rcParams['axes.linewidth'] = 1.2
    pyplot.rcParams['xtick.major.width'] = 1.2
    pyplot.rcParams['ytick.major.width'] = 1.2
    
    plot_variable(
        z_grid, data_lens, 0.0, 1.5, 0.0, 8.0, latent1, latent2,
        os.path.join(plot_folder, f'{tag}/VARIABLE/FIGURE_LENS.pdf'),
    )
    plot_variable(
        z_grid, data_source, 0.0, 1.8, 0.0, 8.0, latent1, latent2,
        os.path.join(plot_folder, f'{tag}/VARIABLE/FIGURE_SOURCE.pdf'),
    )
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Plot Variable')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)
