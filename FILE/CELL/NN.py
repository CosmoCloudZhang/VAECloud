import os
import time
import json
import h5py
import numpy
import pyccl
import argparse
from itertools import product
from matplotlib import colors
from matplotlib import pyplot

def plot_data(mean, ell, bin_size, color_list):
    
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
    
    for (m, n) in product(range(bin_size), range(bin_size)):
        
        plot.plot(ell, mean[m, n,:], color = color_list[m + n], linestyle = '-', linewidth = 2.0, label = r'$m = {:.0f} \quad n = {:.0f}$'.format(m + 1, n + 1))
    
    plot.set_xscale('log')
    plot.set_yscale('log')
    plot.legend(loc = 'upper right', fontsize = 15)
    
    plot.set_ylim(2e-14, 2e-9)
    plot.set_xlim(ell.min(), ell.max())
    
    plot.set_xlabel(r'$\ell$')
    plot.set_ylabel(r'$\Bar{C}_\mathrm{g g}^{m n} (\ell)$')
    
    return figure


def plot_sigma(sigma, ell, bin_size, color_list):
    
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
    
    for (m, n) in product(range(bin_size), range(bin_size)):
        
        plot.plot(ell, sigma[m, n,:], color = color_list[m + n], linestyle = '-', linewidth = 2.0, label = r'$m = {:.0f} \quad n = {:.0f}$'.format(m + 1, n + 1))
    
    plot.set_xscale('log')
    plot.set_yscale('linear')
    plot.legend(loc = 'upper right', fontsize = 15)
    
    plot.set_ylim(0.00, 0.06)
    plot.set_xlim(ell.min(), ell.max())
    
    plot.set_xlabel(r'$\ell$')
    plot.set_ylabel(r'$\sigma_\mathrm{g g}^{m n} / \Bar{C}_\mathrm{g g}^{m n} (\ell)$')
    
    return figure


def plot_correlation(data, ell, bin_size, data_size):
    
    figure, plot = pyplot.subplots(nrows = bin_size ** 2, ncols = bin_size ** 2, figsize = (2 * bin_size ** 2, 2 * bin_size ** 2))
    
    correlation = numpy.corrcoef(data.reshape(data_size, bin_size ** 2 * ell.size), rowvar = False)
    
    for (m, n) in product(range(bin_size ** 2), range(bin_size ** 2)):
        
        norm = colors.Normalize(vmin = -1.0, vmax = 1.0)
        
        map = correlation[n * ell.size: (n + 1) * ell.size, m * ell.size: (m + 1) * ell.size]
        
        image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
        
        plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathrm{R} \: [C^{m_1 m_2} (\ell), C^{m_1^\prime m_2^\prime} (\ell^\prime)]$')
    
    return figure


def main(path, name):
    '''
    Main function for the shear-shear angular power spectrum analysis.
    
    Arguments:
        path (str): The path to the base folder.
        name (str): The name of the angular power spectra.
    
    Returns:
        duration (float): The duration of the PCA analysis
    '''
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    os.makedirs(os.path.join(plot_path, '{}/'.format(name)), exist_ok = True)
    
    # Meta
    with open(os.path.join(data_path, 'LENS/INFO.json'), 'r') as file:
        info = json.load(file)
    bin_size = info['META']['BIN_SIZE']
    grid_size = info['META']['GRID_SIZE']
    mock_size = info['META']['MOCK_SIZE']
    data_size = mock_size // 10
    plot_size = mock_size // 100
    
    # Model
    with h5py.File(os.path.join(data_path, 'LENS/DATA.hdf5'), 'r') as file:
        lens_input = numpy.array(file['TEST_INPUT'])
        lens_output = numpy.array(file['TEST_OUTPUT'])
    
    # Cosmology
    cosmology = json.load(open(data_path + 'COSMO/COSMO.json', 'r'))
    cosmology_ccl = pyccl.Cosmology(
        h=cosmology['H'],
        w0=cosmology['W0'],
        wa=cosmology['WA'],
        n_s=cosmology['NS'], 
        A_s=cosmology['AS'],
        m_nu=cosmology['MNU'],
        Neff=cosmology['NEFF'],
        T_CMB=cosmology['TCMB'],
        Omega_b=cosmology['OMEGAB'], 
        Omega_c=cosmology['OMEGAC'],
        Omega_k=cosmology['OMEGAK'],
        matter_power_spectrum = 'halofit',
        transfer_function = 'boltzmann_camb',
        extra_parameters = {'camb': {'kmax': 100, 'lmax': 10000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    power_ccl = cosmology_ccl.get_nonlin_power(name='delta_matter:delta_matter')
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_delta = (z2 - z1) / grid_size
    z_data = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, grid_size)
    
    # ELL
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_data = numpy.geomspace(ell1, ell2, ell_size)
    
    # C_ELL
    bias_data = numpy.ones(grid_size)
    input_data = numpy.zeros((mock_size, bin_size, bin_size, len(ell_data)))
    output_data = numpy.zeros((mock_size, bin_size, bin_size, len(ell_data)))
    
    index = numpy.random.choice(numpy.arange(data_size), plot_size, replace = False)
    for k in index:
        print(k)
        for (m, n) in product(range(bin_size), range(bin_size)):
            
            input_tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology_ccl, dndz=(z_data, lens_input[k, m,:]), bias=(z_data, bias_data), mag_bias=None, has_rsd=False, n_samples=grid_size)
            input_tracer2 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology_ccl, dndz=(z_data, lens_input[k, n,:]), bias=(z_data, bias_data), mag_bias=None, has_rsd=False, n_samples=grid_size)
            input_data[k, m, n] = pyccl.cells.angular_cl(cosmo=cosmology_ccl, tracer1=input_tracer1, tracer2=input_tracer2, ell=ell_data, p_of_k_a=power_ccl, l_limber=-1, limber_max_error=0.01, limber_integration_method='spline')
            
            output_tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology_ccl, dndz=(z_data, lens_output[k, m,:]), bias=(z_data, bias_data), mag_bias=None, has_rsd=False, n_samples=grid_size)
            output_tracer2 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology_ccl, dndz=(z_data, lens_output[k, n,:]), bias=(z_data, bias_data), mag_bias=None, has_rsd=False, n_samples=grid_size)
            output_data[k, m, n] = pyccl.cells.angular_cl(cosmo=cosmology_ccl, tracer1=output_tracer1, tracer2=output_tracer2, ell=ell_data, p_of_k_a=power_ccl, l_limber=-1, limber_max_error=0.01, limber_integration_method='spline')
    
    input_mean = numpy.mean(input_data, axis = 0)
    output_mean = numpy.mean(output_data, axis = 0)
    
    input_sigma = numpy.std(input_data, axis = 0) / input_mean
    output_sigma = numpy.std(output_data, axis = 0) / output_mean
    
    # Figure
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    color_list = pyplot.cm.jet(numpy.linspace(0.0, +1.5, bin_size * bin_size))
    
    # Input Mean
    figure = plot_data(input_mean, ell_data, bin_size, color_list)
    figure.savefig(os.path.join(plot_path, '{}/INPUT_MEAN.pdf'.format(name)))
    pyplot.close(figure)
    
    # Output Mean
    figure = plot_data(output_mean, ell_data, bin_size, color_list)
    figure.savefig(os.path.join(plot_path, '{}/OUTPUT_MEAN.pdf'.format(name)))
    pyplot.close(figure)
    
    # Input Sigma
    figure = plot_sigma(input_sigma, ell_data, bin_size, color_list)
    figure.savefig(os.path.join(plot_path, '{}/INPUT_SIGMA.pdf'.format(name)))
    pyplot.close(figure)
    
    # Output Sigma
    figure = plot_sigma(output_sigma, ell_data, bin_size, color_list)
    figure.savefig(os.path.join(plot_path, '{}/OUTPUT_SIGMA.pdf'.format(name)))
    pyplot.close(figure)
    
    # Input Correlation
    figure = plot_correlation(input_data, ell_data, bin_size, data_size)
    figure.savefig(os.path.join(plot_path, '{}/INPUT_CORRELATION.pdf'.format(name)))
    pyplot.close(figure)
    
    # Output Correlation
    figure = plot_correlation(output_data, ell_data, bin_size, data_size)
    figure.savefig(os.path.join(plot_path, '{}/OUTPUT_CORRELATION.pdf'.format(name)))
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='NN')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the angular power spectra')
    
    PATH = PARSE.parse_args().path
    NAME = PARSE.parse_args().name
    RESULT = main(PATH, NAME)