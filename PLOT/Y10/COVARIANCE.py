import os
import time
import h5py
import numpy
import argparse
from matplotlib import colors, pyplot


def plot_covariance(bin_size, map_size, covariance):
    '''
    Plot the covariance matrix
    
    Arguments:
        bin_size (int): The number of bins
        map_size (int): The size of the map
        covariance (numpy.ndarray): The covariance matrix
    
    Returns:
        figure (matplotlib.figure.Figure): The figure
    '''
    norm = colors.Normalize(vmin = -0.5, vmax = +0.5)
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    for m in range(bin_size):
        for n in range(bin_size):
            
            map_covariance = covariance[n * map_size: (n + 1) * map_size, m * map_size: (m + 1) * map_size]
            image = plot[n, m].imshow(map_covariance, norm = norm, cmap = 'seismic', origin = 'upper', rasterized = True)
            
            plot[n, m].set_xticks([])
            plot[n, m].set_yticks([])
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.0, hspace = 0.0)
    
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathcal{C} \: [\phi^{i} (z_1), \phi^{j} (z_2)]$')
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
    data_folder = os.path.join(folder, 'DATA/')
    plot_folder = os.path.join(folder, 'PLOT/')
    lens_folder = os.path.join(folder, 'LENS/')
    source_folder = os.path.join(folder, 'SOURCE/')
    os.makedirs(os.path.join(plot_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(plot_folder, '{}/COVARIANCE/'.format(tag)), exist_ok = True)
    
    # Correct
    with h5py.File(os.path.join(data_folder, '{}/CORRECT.hdf5'.format(tag)), 'r') as file:
        z1 = file['meta']['z1'][...]
        z2 = file['meta']['z2'][...]
        
        bin_lens = file['meta']['bin_lens'][...]
        bin_lens_size = int(file['meta']['bin_lens_size'][...])
        
        bin_source = file['meta']['bin_source'][...]
        bin_source_size = int(file['meta']['bin_source_size'][...])
    
    # Lens
    with h5py.File(os.path.join(lens_folder, '{}/VAE.hdf5'.format(tag)), 'r') as file:
        input_data_lens = file['data']['input'][...]
        output_data_lens = file['data']['output'][...]
        
        z_size = int(file['meta']['z_size'][...])
        selection_size = file['meta']['selection_size'][...]
        evaluation_size = file['meta']['evaluation_size'][...]
        data_lens_size = int(file['meta']['data_size'][...] * selection_size * evaluation_size)
    
    # Source
    with h5py.File(os.path.join(source_folder, '{}/VAE.hdf5'.format(tag)), 'r') as file:
        input_data_source = file['data']['input'][...]
        output_data_source = file['data']['output'][...]
        
        z_size = int(file['meta']['z_size'][...])
        selection_size = file['meta']['selection_size'][...]
        evaluation_size = file['meta']['evaluation_size'][...]
        data_source_size = int(file['meta']['data_size'][...] * selection_size * evaluation_size)
    
    # Size
    map_lens_size = 40
    map_source_size = 100
    z_delta = (z2 - z1) / (z_size - 1)
    
    # Index
    index_lower_lens = numpy.maximum(0, numpy.array(numpy.round(bin_lens[:-1] / z_delta, decimals=0), dtype='int32') - map_lens_size // 4)
    index_upper_lens = index_lower_lens + map_lens_size
    
    index_lower_source = numpy.maximum(0, numpy.array(numpy.round(bin_source[:-1] / z_delta, decimals=0), dtype='int32') - map_source_size // 4)
    index_upper_source = index_lower_source + map_source_size
    
    # Select
    input_select_lens = numpy.zeros((data_lens_size, bin_lens_size, map_lens_size))
    output_select_lens = numpy.zeros((data_lens_size, bin_lens_size, map_lens_size))
    input_select_source = numpy.zeros((data_source_size, bin_source_size, map_source_size))
    output_select_source = numpy.zeros((data_source_size, bin_source_size, map_source_size))
    
    for m in range(bin_lens_size):
        input_select_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = input_data_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
        output_select_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = output_data_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
    
    for m in range(bin_source_size):
        input_select_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = input_data_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
        output_select_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = output_data_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
    
    input_covariance_lens = numpy.cov(numpy.reshape(input_select_lens, (data_lens_size, bin_lens_size * map_lens_size)), rowvar=False)
    output_covariance_lens = numpy.cov(numpy.reshape(output_select_lens, (data_lens_size, bin_lens_size * map_lens_size)), rowvar=False)
    input_covariance_source = numpy.cov(numpy.reshape(input_select_source, (data_source_size, bin_source_size * map_source_size)), rowvar=False)
    output_covariance_source = numpy.cov(numpy.reshape(output_select_source, (data_source_size, bin_source_size * map_source_size)), rowvar=False)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Plot
    figure = plot_covariance(bin_lens_size, map_lens_size, input_covariance_lens)
    figure.savefig(os.path.join(plot_folder, '{}/COVARIANCE/FIGURE_INPUT_LENS.pdf'.format(tag)), format='pdf', bbox_inches='tight', dpi = 512)
    pyplot.close(figure)
    
    figure = plot_covariance(bin_lens_size, map_lens_size, output_covariance_lens)
    figure.savefig(os.path.join(plot_folder, '{}/COVARIANCE/FIGURE_OUTPUT_LENS.pdf'.format(tag)), format='pdf', bbox_inches='tight', dpi = 512)
    pyplot.close(figure)
    
    figure = plot_covariance(bin_source_size, map_source_size, input_covariance_source)
    figure.savefig(os.path.join(plot_folder, '{}/COVARIANCE/FIGURE_INPUT_SOURCE.pdf'.format(tag)), format='pdf', bbox_inches='tight', dpi = 512)
    pyplot.close(figure)
    
    figure = plot_covariance(bin_source_size, map_source_size, output_covariance_source)
    figure.savefig(os.path.join(plot_folder, '{}/COVARIANCE/FIGURE_OUTPUT_SOURCE.pdf'.format(tag)), format='pdf', bbox_inches='tight', dpi = 512)
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Plot Covariance')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)
