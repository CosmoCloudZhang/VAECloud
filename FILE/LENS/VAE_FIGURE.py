
def plot_loss():
    FIGURE, PLOT = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 12))

    EPOCH = numpy.arange(EPOCH_SIZE, dtype = 'int') + 1

    PLOT.plot(EPOCH, HISTORY.history['all_loss'], color = 'black', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{all}$')

    PLOT.plot(EPOCH, HISTORY.history['mean_loss'], color = 'purple', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{mean}$')

    PLOT.plot(EPOCH, HISTORY.history['scatter_loss'], color = 'blue', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{scatter}$')

    PLOT.plot(EPOCH, HISTORY.history['covariance_loss'], color = 'green', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{covariance}$')

    PLOT.plot(EPOCH, HISTORY.history['divergence_loss'], color = 'orange', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{divergence}$')

    PLOT.set_yscale('log')
    PLOT.set_xlim(1, EPOCH_SIZE)

    PLOT.legend()
    PLOT.set_xlabel(r'$\mathrm{Epochs}$')
    PLOT.set_ylabel(r'$\mathrm{Losses}$')

    FIGURE.savefig(PLOT_PATH + 'LOSS.pdf', bbox_inches = 'tight')
    return figure


def plot_data():
    FIGURE, PLOT = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
    
    for m in range(BIN_SIZE):
        
        PHI_QUANTILE0 = numpy.quantile(TEST_OUTPUT[:, m, :], q=0.50, axis=0)
        PHI_QUANTILE1 = numpy.quantile(TEST_OUTPUT[:, m, :], q=0.16, axis=0)
        PHI_QUANTILE2 = numpy.quantile(TEST_OUTPUT[:, m, :], q=0.84, axis=0)
        
        PLOT.plot(Z_GRID, PHI_QUANTILE0, color = COLOR[m], linewidth = 2.0, label = r'$m = {:.0f}$'.format(m + 1))

        PLOT.fill_between(Z_GRID, PHI_QUANTILE1, PHI_QUANTILE2, color = COLOR[m], alpha = 0.25)

    PLOT.legend()
    PLOT.set_xlim(Z1, Z2)
    PLOT.set_ylim(PHI1, PHI2)

    PLOT.set_xlabel(r'$z$')
    PLOT.set_ylabel(r'$\phi^m (z)$')

    FIGURE.savefig(PLOT_PATH + 'OUTPUT_DATA.pdf', bbox_inches = 'tight')
    return figure

def plot_covariance():
    
    FIGURE, PLOT = pyplot.subplots(nrows = BIN_SIZE, ncols = BIN_SIZE, figsize = (3 * BIN_SIZE, 3 * BIN_SIZE))

    INPUT_COVARIANCE = numpy.cov(TEST_INPUT.reshape(TEST_SIZE, BIN_SIZE * GRID_SIZE), rowvar = False)

    for m in range(BIN_SIZE):
        
        for n in range(BIN_SIZE):
            
            NORM = colors.SymLogNorm(vmin = -0.01, vmax = +0.5, linthresh = +0.01)
            
            MAP = INPUT_COVARIANCE[n * GRID_SIZE: (n + 1) * GRID_SIZE, m * GRID_SIZE: (m + 1) * GRID_SIZE]
            
            IMAGE = PLOT[n, m].imshow(MAP, norm = NORM, cmap = 'viridis', origin = 'upper')
            
            PLOT[n, m].axis('off')
            
    CAX = FIGURE.add_axes([0.15, 0.05, 0.72, 0.05])
    COLORBAR = FIGURE.colorbar(cax = CAX, mappable = IMAGE, orientation = 'horizontal', label = '$\mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')

    FIGURE.subplots_adjust(wspace = 0.02, hspace = 0.02)
    FIGURE.savefig(PLOT_PATH + 'INPUT_COVARIANCE.pdf', bbox_inches = 'tight')
    return FIGURE