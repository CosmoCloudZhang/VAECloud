import os
import time
import h5py
import numpy
import scipy
import argparse
import scipy.interpolate
from sklearn import decomposition
from matplotlib import pyplot, colors

def plot_pca(phi, bin_size, grid_size):
    '''
    Plot the PCA.
    
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
        
        plot.plot(z_data, phi[m, :], color = color_list[m], linewidth = 2.5)
    
    plot.set_ylim(0.0, 0.25)
    plot.set_xlim(z1, z2 * 2 / 3)
    
    plot.set_xticklabels([])
    plot.set_yticklabels([])
    
    return figure


def plot_variance(model, latent_size):
    '''
    Plot the variance.
    
    Arguments:
        model (sklearn.decomposition.PCA): The PCA model.
        latent_size (int): The latent size.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    '''
    figure, plot = pyplot.subplots(nrows = 2, ncols = 1, figsize = (10, 10))
    
    plot[0].plot(numpy.arange(1, latent_size + 1), model.explained_variance_ratio_, color = 'black', linestyle = '-', linewidth = 2.5)
    
    plot[0].set_ylim(0.00, 0.20)
    plot[0].set_xlim(1, latent_size)
    
    ratio = numpy.array([model.explained_variance_ratio_[:n + 1].sum() for n in range(latent_size)])
    
    plot[1].plot(numpy.arange(1, latent_size + 1), ratio, color = 'black', linestyle = '-', linewidth = 2.5)
    
    plot[1].set_ylim(0, 1)
    plot[1].set_xlim(1, latent_size)
    
    return figure


def plot_shift(phi):
    
    # Shift
    z1 = 0.0
    z2 = 3.0
    z_size = 300
    z_delta = (z2 - z1) / z_size
    z_grid = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, z_size)
    
    z_mean = numpy.sum(phi * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    correlation = numpy.corrcoef(z_mean, rowvar=False)
    
    mean = numpy.mean(z_mean, axis=0)
    scatter = numpy.std(z_mean, axis=0)
    
    print(mean)
    print(scatter)
    
    # Plot
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
    
    norm = colors.Normalize(vmin = -1.0, vmax = +1.0)
    image = plot.imshow(correlation, norm = norm, cmap = 'coolwarm', origin = 'upper')
    
    for (i, j), value in numpy.ndenumerate(correlation):
        plot.text(j, i, r'${:.3f}$'.format(value), va='center', ha='center', color='black')
    plot.axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathrm{Cov} \: [\Delta z^{m_1}, \Delta z^{m_2}]$')
    
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    return figure


def plot_delta(phi, mock_size, bin_size, grid_size):
    
    # Shift
    z1 = 0.0
    z2 = 3.0
    z_delta = (z2 - z1) / grid_size
    z_grid = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, grid_size)
    
    z_mean = numpy.sum(phi * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    covariance = numpy.cov(z_mean, rowvar=False)
    
    # Phi
    shift = numpy.random.multivariate_normal(mean=numpy.zeros(bin_size), cov=covariance, size=mock_size)
    z_shift = z_grid[numpy.newaxis, numpy.newaxis, :] + shift[:, :, numpy.newaxis]
    
    phi_mean = numpy.mean(phi, axis=0)
    phi_shift = numpy.zeros((mock_size, bin_size, grid_size))
    for m in range(bin_size):
        phi_shift[:, m, :] = scipy.interpolate.CubicSpline(z_grid, phi_mean[m, :], extrapolate=True)(z_shift[:, m, :])
    
    # Delta
    delta = numpy.cov(numpy.reshape(phi, (-1, bin_size * grid_size)), rowvar=False)
    delta = delta - numpy.cov(numpy.reshape(phi_shift, (-1, bin_size * grid_size)), rowvar=False)
    
    # Plot
    norm = colors.Normalize(vmin = -0.05, vmax = +0.05)
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    for m in range(bin_size):
        for n in range(bin_size):
            
            map = delta[n * grid_size: n * grid_size + grid_size // 2, m * grid_size: m * grid_size + grid_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\Delta \mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    
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
    os.makedirs(os.path.join(plot_path, '{}/PCA/'.format(name)), exist_ok = True)
    
    # Data
    data_name = os.path.join(data_path, '{}/INPUT.hdf5'.format(name))
    with h5py.File(data_name, 'r') as file:
        input_sample = file['sample'][:].astype(numpy.float64)
    mock_size, bin_size, grid_size = input_sample.shape
    
    # Input
    delta = 0.01
    input_data = numpy.maximum(input_sample, 0.0)
    input_data = input_data / numpy.sum(input_data, axis=2, keepdims=True) / delta
    
    # PCA
    latent_size = 128
    pca_model = decomposition.PCA(n_components = latent_size)
    
    latent_data = pca_model.fit_transform(numpy.reshape(input_data, (mock_size, bin_size * grid_size)))
    output_data = numpy.reshape(pca_model.inverse_transform(latent_data), (mock_size, bin_size, grid_size))
    
    pca_data = numpy.reshape(pca_model.components_, (latent_size, bin_size, grid_size))
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    pca_size = 5
    for n in range(pca_size):
        figure = plot_pca(pca_data[n, :, :], bin_size, grid_size)
        figure.savefig(os.path.join(plot_path, '{}/PCA/PCA_{:.0f}.pdf'.format(name, n + 1)), bbox_inches = 'tight')
        pyplot.close(figure)
    
    figure = plot_variance(pca_model, latent_size)
    figure.savefig(os.path.join(plot_path, '{}/PCA/VARIANCE.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = plot_shift(input_data)
    figure.savefig(os.path.join(plot_path, '{}/PCA/SHIFT.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = plot_delta(output_data, mock_size, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/PCA/DELTA.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='PCA')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the sample datasets')
    
    PATH = PARSE.parse_args().path
    NAME = PARSE.parse_args().name
    RESULT = main(PATH, NAME)