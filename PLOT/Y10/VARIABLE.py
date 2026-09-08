import argparse
import os
import pickle
import time

import h5py
import numpy
from matplotlib import cm, colors, pyplot, ticker
from tensorflow import keras


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
    
    # Lens
    figure, plot = pyplot.subplots(nrows=latent_lens_size, ncols=1, figsize=(12, 3 * latent_lens_size), sharex=True)
    if latent_lens_size == 1:
        plot = [plot]
    
    norm = colors.Normalize(vmin=latent1, vmax=latent2)
    scalar_mappable = cm.ScalarMappable(cmap=cm.plasma_r, norm=norm)
    color_list = cm.plasma_r(numpy.linspace(0.0, 1.0, sample_size + 1))
    
    for k in range(latent_lens_size):
        for n in range(sample_size + 1):
            for m in range(bin_lens_size):
                plot[k].plot(z_grid, data_lens[k, n, m, :], color = color_list[n], linewidth = 1.0, rasterized = True)
        plot[k].set_xlim(0.0, 1.4)
        plot[k].set_ylim(0.0, 8.0)
        plot[k].set_ylabel(r'$\phi^m (z)$')
        plot[k].text(0.02, 0.82, rf'$\xi_{{{k + 1}}}$ in $[{latent1:.0f},{latent2:+.0f}]$, others $0$', transform=plot[k].transAxes, fontsize=16)
        plot[k].yaxis.set_major_locator(ticker.FixedLocator(numpy.arange(0.0, 8.0, 2.0)))
    
    plot[-1].set_xlabel(r'$z$')
    figure.colorbar(mappable=scalar_mappable, ax=plot, orientation='vertical', label=r'$\xi_k$', shrink=0.9)
    figure.subplots_adjust(wspace = 0.05, hspace = 0.05)
    
    figure.savefig(os.path.join(plot_folder, f'{tag}/VARIABLE/FIGURE_LENS.pdf'), format='pdf', bbox_inches='tight', dpi=512)
    pyplot.close(figure)
    
    # Source
    figure, plot = pyplot.subplots(nrows=latent_source_size, ncols=1, figsize=(12, 3 * latent_source_size), sharex=True)
    if latent_source_size == 1:
        plot = [plot]
    
    norm = colors.Normalize(vmin=latent1, vmax=latent2)
    scalar_mappable = cm.ScalarMappable(cmap=cm.plasma_r, norm=norm)
    color_list = cm.plasma_r(numpy.linspace(0.0, 1.0, sample_size + 1))
    
    for k in range(latent_source_size):
        for n in range(sample_size + 1):
            for m in range(bin_source_size):
                plot[k].plot(z_grid, data_source[k, n, m, :], color = color_list[n], linewidth = 1.0, rasterized = True)
        plot[k].set_xlim(0.0, 2.1)
        plot[k].set_ylim(0.0, 8.0)
        plot[k].set_ylabel(r'$\phi^m (z)$')
        plot[k].text(0.02, 0.82, rf'$\xi_{{{k + 1}}}$ in $[{latent1:.0f},{latent2:+.0f}]$, others $0$', transform=plot[k].transAxes, fontsize=16)
        plot[k].yaxis.set_major_locator(ticker.FixedLocator(numpy.arange(0.0, 8.0, 2.0)))
    
    plot[-1].set_xlabel(r'$z$')
    figure.colorbar(mappable=scalar_mappable, ax=plot, orientation='vertical', label=r'$\xi_k$', shrink=0.9)
    figure.subplots_adjust(wspace = 0.05, hspace = 0.05)
    
    figure.savefig(os.path.join(plot_folder, f'{tag}/VARIABLE/FIGURE_SOURCE.pdf'), format='pdf', bbox_inches='tight', dpi=512)
    pyplot.close(figure)
    
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
