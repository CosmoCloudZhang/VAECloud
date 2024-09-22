import numpy
from matplotlib import colors
from matplotlib import pyplot


def plot_data(phi, z_grid):
    """
    Plot the data.
    
    Arguments:
        phi (numpy.ndarray): The data.
        z_grid (numpy.ndarray): The grid.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    """
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
    """
    Plot the variance.
    
    Arguments:
        model (sklearn.decomposition.PCA): The PCA model.
        latent_size (int): The latent size.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    """
    figure, plot = pyplot.subplots(nrows = 2, ncols = 1, figsize = (10, 10))
    
    plot[0].plot(numpy.arange(1, latent_size + 1), model.explained_variance_ratio_, color = 'black', linestyle = '-', linewidth = 2.5)
    
    plot[0].set_ylim(0.00, 0.20)
    plot[0].set_xlim(1, latent_size)
    plot[0].set_title(r'$\sum \frac{\sigma_i^2}{\sigma^2} = ' + '{:.0f}'.format(model.explained_variance_ratio_.sum() * 100) + '\%$')
    
    ratio = numpy.array([model.explained_variance_ratio_[:n + 1].sum() for n in range(latent_size)])
    
    plot[1].plot(numpy.arange(1, latent_size + 1), ratio, color = 'black', linestyle = '-', linewidth = 2.5)
    
    plot[1].set_ylim(0, 1)
    plot[1].set_xlim(1, latent_size)
    
    return figure


def plot_covariance(phi, bin_size, grid_size):
    """
    Plot the covariance.
    
    Arguments:
        phi (numpy.ndarray): The data.
        bin_size (int): The bin size.
        grid_size (int): The grid size.
    
    Returns:
        figure (matplotlib.figure.Figure): The figure.
    """
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    covariance = numpy.cov(phi, rowvar = False)
    norm = colors.Normalize(vmin = -0.1, vmax = +0.1)
    
    for m in range(bin_size):
        for n in range(bin_size):
            
            map = covariance[n * grid_size: n * grid_size + grid_size // 2, m * grid_size: m * grid_size + grid_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    
    return figure