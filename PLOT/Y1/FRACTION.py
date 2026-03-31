import os
import time
import numpy
import pickle
import argparse
from matplotlib import pyplot


def main(tag, folder):
    '''
    Plot the fractional variance as function of the pca size
    
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
    os.makedirs(os.path.join(plot_folder, '{}/FRACTION/'.format(tag)), exist_ok = True)
    
    # Load
    with open(os.path.join(lens_folder, '{}/PCA.pkl'.format(tag)), 'rb') as file:
        pca_lens_model = pickle.load(file)
    pca_lens_size = pca_lens_model.n_components_
    pca_lens_indices = numpy.arange(1, pca_lens_size + 1)
    pca_lens_fraction = pca_lens_model.explained_variance_ratio_
    
    with open(os.path.join(source_folder, '{}/PCA.pkl'.format(tag)), 'rb') as file:
        pca_source_model = pickle.load(file)
    pca_source_size = pca_source_model.n_components_
    pca_source_indices = numpy.arange(1, pca_source_size + 1)
    pca_source_fraction = pca_source_model.explained_variance_ratio_
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Figure
    figure, plot = pyplot.subplots(nrows = 2, ncols = 1, figsize = (10, 10))
    
    plot[0].plot(pca_lens_indices, pca_lens_fraction, color = 'darkblue', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{Lens}$', rasterized = True)
    
    plot[0].plot(pca_source_indices, pca_source_fraction, color = 'darkred', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{Source}$', rasterized = True)
    
    plot[0].set_yscale('log')
    plot[0].set_xticklabels([])
    plot[0].set_ylim(5e-6, 1e+0)
    
    plot[0].set_xlim(1, pca_lens_size)
    plot[0].set_xlim(1, pca_source_size)
    
    plot[0].legend(loc='upper right')
    plot[0].set_ylabel(r'$f_{\sigma^2}(n)$')
    
    pca_lens_cumulative = numpy.array([pca_lens_fraction[:n + 1].sum() for n in range(pca_lens_size)])
    pca_source_cumulative = numpy.array([pca_source_fraction[:n + 1].sum() for n in range(pca_source_size)])
    
    plot[1].plot(pca_lens_indices, pca_lens_cumulative, color = 'darkblue', linestyle = '-', linewidth = 2.5, rasterized = True)
    
    plot[1].plot(pca_source_indices, pca_source_cumulative, color = 'darkred', linestyle = '-', linewidth = 2.5, rasterized = True)
    
    plot[1].set_ylim(0, 1)
    plot[1].set_xlim(1, pca_lens_size)
    plot[1].set_xlim(1, pca_source_size)
    
    plot[1].set_xlabel(r'$n$')
    plot[1].set_ylabel(r'$\sum_{i}^{n} f_\mathrm{\sigma^2}(i)$')
    
    figure.subplots_adjust(hspace = 0.0, wspace = 0.0)
    figure.savefig(os.path.join(plot_folder, '{}/FRACTION/FIGURE.pdf'.format(tag)), format = 'pdf', bbox_inches = 'tight', dpi = 512)
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Plot Fraction')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)
