import os
import time
import numpy
import pickle
import argparse
from matplotlib import pyplot


def main(tag, folder):
    '''
    Plot the variance as function of the pca size
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The folder containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/{}/'.format(tag))
    pca_lens_folder = os.path.join(folder, 'LENS/{}/PCA/'.format(tag))
    pca_source_folder = os.path.join(folder, 'SOURCE/{}/PCA/'.format(tag))
    
    # Load
    with open(os.path.join(pca_lens_folder, 'MODEL.pkl'), 'rb') as file:
        pca_lens_model = pickle.load(file)
    pca_lens_variance = pca_lens_model.explained_variance_ratio_
    
    with open(os.path.join(pca_source_folder, 'MODEL.pkl'), 'rb') as file:
        pca_source_model = pickle.load(file)
    pca_source_variance = pca_source_model.explained_variance_ratio_
    
    pca_lens_size = pca_lens_model.n_components_
    pca_source_size = pca_source_model.n_components_
    pca_size = (pca_lens_size + pca_source_size) // 2
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    
    z_grid = numpy.linspace(z1, z2, data_size)
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    bin_size = 5
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
    color_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    
    for m in range(bin_size):
        
        plot.plot(z_grid, phi[m, :], color = color_list[m], linewidth = 2.5)
    
    plot.set_xlim(z1, z2)
    plot.set_ylim(0.0, 0.30)
    
    plot.set_xticklabels([])
    plot.set_yticklabels([])
    os.makedirs(os.path.join(figure_folder, 'COMPONENT/'), exist_ok = True)
    figure.savefig(os.path.join(figure_folder, 'COMPONENT/FIGURE_LENS.pdf'), format = 'pdf', bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Component')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)

