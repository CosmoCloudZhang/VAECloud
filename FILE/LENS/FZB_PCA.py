import os
import time
import h5py
import numpy
import argparse
from matplotlib import colors
from matplotlib import pyplot
from sklearn import decomposition


def plot_data(phi, z_grid):
    
    z_grid = (z_grid[+1:] + z_grid[:-1]) / 2
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
    color_list = ['darkmagenta', 'darkblue', 'darkcyan', 'darkorange', 'darkred']
    
    for m in range(len(color_list)):
        
        phi0 = numpy.mean(phi[:, m, :], axis = 0)
        
        plot.plot(z_grid, phi0, color = color_list[m], linewidth = 2.0, label = r'$m = {:.0f}$'.format(m + 1))
        
        plot.plot(z_grid, numpy.transpose(phi[:, m, :]), color = color_list[m], linewidth = 0.005, alpha = 0.005)
    
    plot.legend()
    plot.set_ylim(0.0, 8.0)
    plot.set_xlim(z_grid.min(), z_grid.max())
    
    plot.set_xlabel(r'$z$')
    plot.set_ylabel(r'$\phi^m (z)$')
    
    return figure


def plot_variance(model, latent_size):
    
    figure, plot = pyplot.subplots(nrows = 2, ncols = 1, figsize = (10, 10))
    
    plot[0].plot(numpy.arange(1, latent_size + 1), model.explained_variance_ratio_, color = 'black', linestyle = '-', linewidth = 2.5)
    
    plot[0].set_ylim(0.00, 0.20)
    plot[0].set_xlim(1, latent_size + 1)
    plot[0].set_title(r'$\sum \frac{\sigma_i^2}{\sigma^2} = ' + '{:.0f}'.format(model.explained_variance_ratio_.sum() * 100) + '\%$')
    
    ratio = numpy.array([model.explained_variance_ratio_[:n + 1].sum() for n in range(latent_size)])
    
    plot[1].plot(numpy.arange(1, latent_size + 1), ratio, color = 'black', linestyle = '-', linewidth = 2.5)
    
    plot[1].set_ylim(0, 1)
    plot[1].set_xlim(1, latent_size)
    
    return figure


def plot_covariance(phi, bin_size, grid_size):
    
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    covariance = numpy.cov(phi, rowvar = False)
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


def main(path, name, label):
    """
    Main function for the PCA analysis.
    
    Arguments:
        path (str): The path to the base folder.
        name (str): The name of the sample datasets.
    
    Returns:
        duration (float): The duration of the PCA analysis
    """
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    
    # Data
    data_name = os.path.join(data_path, '{}/{}/INPUT.hdf5'.format(name, label))
    with h5py.File(data_name, 'r') as file:
        input = file['sample'][:].astype(numpy.float32)
    
    mock_size, bin_size, grid_size = input.shape
    data_size = bin_size * grid_size
    latent_size = 64
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Model
    input_data = input.reshape(mock_size, data_size)
    model = decomposition.PCA(n_components = latent_size)
    
    latent_data = model.fit_transform(input_data)
    output_data = model.inverse_transform(latent_data)
    output = output_data.reshape(mock_size, bin_size, grid_size)
    
    # Plot
    plot_size = mock_size // 40
    select = numpy.random.choice(numpy.arange(mock_size), plot_size, replace = False)
    
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    figure = plot_data(input[select, :, :], z_grid)
    figure.savefig(os.path.join(plot_path, '{}/{}/PCA/INPUT_DATA.pdf'.format(name, label)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = plot_data(output[select, :, :], z_grid)
    figure.savefig(os.path.join(plot_path, '{}/{}/PCA/OUTPUT_DATA.pdf'.format(name, label)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = plot_variance(model, latent_size)
    figure.savefig(os.path.join(plot_path, '{}/{}/PCA/VARIANCE.pdf'.format(name, label)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = plot_covariance(input_data, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/{}/PCA/INPUT_COVARIANCE.pdf'.format(name, label)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = plot_covariance(output_data, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/{}/PCA/OUTPUT_COVARIANCE.pdf'.format(name, label)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB PCA')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the sample datasets')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the method for photometric redshifts')
    
    PATH = PARSE.parse_args().path
    NAME = PARSE.parse_args().name
    LABEL = PARSE.parse_args().label
    RESULT = main(PATH, NAME, LABEL)