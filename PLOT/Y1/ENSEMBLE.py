import os
import time
import h5py
import numpy
import argparse
from matplotlib import pyplot


def plot_ensemble(z_grid, select_lens, select_source, bin_lens_size, bin_source_size):
    '''
    Plot the ensemble of the lens and source redshift distributions
    
    Arguments:
        z_grid (numpy.ndarray): The redshift grid
        select_lens (numpy.ndarray): The lens distributions
        select_source (numpy.ndarray): The source distributions
        bin_lens_size (int): The number of lens tomographic bins
        bin_source_size (int): The number of source tomographic bins
    
    Returns:
        figure (matplotlib.figure.Figure): The figure
    '''
    # Figure
    figure, plot = pyplot.subplots(nrows = 2, ncols = 1, figsize = (12, 12))
    color_lens_list = ['darkmagenta', 'darkblue', 'darkcyan', 'darkorange', 'darkred']
    color_source_list = ['darkmagenta', 'darkblue', 'darkcyan', 'darkorange', 'darkred']
    
    for m in range(bin_lens_size):
        
        plot[0].plot(z_grid, numpy.transpose(select_lens[:, m, :]), color = color_lens_list[m], linewidth = 0.1, alpha = 0.1, rasterized=True)
        
        plot[0].plot(z_grid, numpy.mean(select_lens[:, m, :], axis=0), color = color_lens_list[m], linewidth = 2.5, label=r'$\mathrm{Bin} \,' + r'{:.0f}$'.format(m + 1), rasterized=True)
    
    plot[0].set_xlim(0.0, 2.0)
    plot[0].set_ylim(0.0, 8.0)
    
    plot[0].set_xticklabels([])
    plot[0].set_yticks([2.0, 4.0, 6.0, 8.0])
    plot[0].legend(loc='upper right', fontsize=20)
    
    plot[0].set_ylabel(r'$\varphi_m \left( z \right)$')
    plot[0].text(x=1.6, y=3.0, s=r'$\mathrm{Lens}$', fontsize=25)
    
    for m in range(bin_source_size):
        
        plot[1].plot(z_grid, numpy.transpose(select_source[:, m, :]), color = color_source_list[m], linewidth = 0.1, alpha = 0.1, rasterized=True)
        
        plot[1].plot(z_grid, numpy.mean(select_source[:, m, :], axis=0), color = color_source_list[m], linewidth = 2.5, label=r'$\mathrm{Bin} \,' + r'{:.0f}$'.format(m + 1), rasterized=True)
    
    plot[1].set_xlim(0.0, 2.0)
    plot[1].set_ylim(0.0, 8.0)
    
    plot[1].set_yticks([0.0, 2.0, 4.0, 6.0, 8.0])
    plot[1].legend(loc='upper right', fontsize=20)
    
    plot[1].set_xlabel(r'$z$')
    plot[1].set_ylabel(r'$\varphi_m \left( z \right)$')
    plot[1].text(x=1.6, y=3.0, s=r'$\mathtt{Source}$', fontsize=25)
    
    figure.subplots_adjust(hspace=0.0)
    return figure


def main(tag, folder):
    '''
    Plot the loss of the VAE process
    
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
    os.makedirs(os.path.join(plot_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(plot_folder, '{}/ENSEMBLE/'.format(tag)), exist_ok = True)
    
    # Lens
    with h5py.File(os.path.join(lens_folder, '{}/VAE.hdf5'.format(tag)), 'r') as file:
        input_data_lens = file['data']['input'][...]
        output_data_lens = file['data']['output'][...]
        
        z_size = int(file['meta']['z_size'][...])
        bin_lens_size = int(file['meta']['bin_size'][...])
        data_lens_size = int(file['meta']['data_size'][...])
        
        selection_size = file['meta']['selection_size'][...]
        evaluation_size = file['meta']['evaluation_size'][...]
        sample_lens_size = int(data_lens_size * selection_size * evaluation_size)
    
    # Source
    with h5py.File(os.path.join(source_folder, '{}/VAE.hdf5'.format(tag)), 'r') as file:
        input_data_source = file['data']['input'][...]
        output_data_source = file['data']['output'][...]
        
        z_size = int(file['meta']['z_size'][...])
        bin_source_size = int(file['meta']['bin_size'][...])
        data_source_size = int(file['meta']['data_size'][...])
        
        selection_size = file['meta']['selection_size'][...]
        evaluation_size = file['meta']['evaluation_size'][...]
        sample_source_size = int(data_source_size * selection_size * evaluation_size)
    
    # Size
    select_size = 10000
    indices_lens = numpy.sort(numpy.random.choice(sample_lens_size, select_size, replace=False))
    indices_source = numpy.sort(numpy.random.choice(sample_source_size, select_size, replace=False))
    
    # Select
    input_select_lens = input_data_lens[indices_lens, :, :]
    output_select_lens = output_data_lens[indices_lens, :, :]
    
    input_select_source = input_data_source[indices_source, :, :]
    output_select_source = output_data_source[indices_source, :, :]
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, z_size)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Input
    figure = plot_ensemble(z_grid, input_select_lens, input_select_source, bin_lens_size, bin_source_size)
    figure.savefig(os.path.join(plot_folder, '{}/ENSEMBLE/FIGURE_INPUT.pdf'.format(tag)), bbox_inches='tight', dpi = 512)
    pyplot.close(figure)
    
    # Output
    figure = plot_ensemble(z_grid, output_select_lens, output_select_source, bin_lens_size, bin_source_size)
    figure.savefig(os.path.join(plot_folder, '{}/ENSEMBLE/FIGURE_OUTPUT.pdf'.format(tag)), bbox_inches='tight', dpi = 512)
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Plot Ensemble')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)
