import os
import json
import time
import argparse
from itertools import product

import h5py
import numpy
import pyccl
import scipy


def main(tag, folder):
    '''
    Calculate the shape-shape angular power spectra
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print(f'Tag: {tag}')
    
    # Path
    cell_folder = os.path.join(folder, 'CELL/')
    info_folder = os.path.join(folder, 'INFO/')
    data_folder = os.path.join(folder, 'DATA/')
    os.makedirs(os.path.join(cell_folder, f'{tag}/'), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, f'{tag}/EE/'), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(data_folder, f'{tag}/CORRECT.hdf5'), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta']}
        average_source = file['source']['average'][...]
        data_source = file['source']['data'][...]
    
    # Meta
    z_grid = meta['z_grid']
    data_size = int(meta['data_size'][...])
    grid_size = int(meta['grid_size'][...])
    bin_source_size = int(meta['bin_source_size'][...])
    
    # Sample
    seed = 1
    numpy.random.seed(seed)
    sample_size = int(data_size * 0.2 * 0.5)
    sample_index = numpy.random.choice(data_size, size=sample_size, replace=False)
    
    # Cosmology
    with open(os.path.join(info_folder, 'COSMOLOGY.json'), 'r') as file:
        cosmology_info = json.load(file)
    
    cosmology = pyccl.Cosmology(
        h=cosmology_info['H'],
        w0=cosmology_info['W0'],
        wa=cosmology_info['WA'],
        n_s=cosmology_info['NS'], 
        A_s=cosmology_info['AS'], 
        m_nu=cosmology_info['M_NU'],
        Neff=cosmology_info['N_EFF'],
        Omega_k=cosmology_info['OMEGA_K'], 
        Omega_b=cosmology_info['OMEGA_B'], 
        Omega_c=cosmology_info['OMEGA_CDM'],
        Omega_g=cosmology_info['OMEGA_GAMMA'], 
        mass_split='single', matter_power_spectrum='halofit', transfer_function='boltzmann_camb',
        extra_parameters={'camb': {'kmax': 50, 'lmax': 5000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    pyccl.gsl_params['NZ_NORM_SPLINE_INTEGRATION'] = False
    pyccl.gsl_params['LENSING_KERNEL_SPLINE_INTEGRATION'] = False
    
    pyccl.gsl_params['INTEGRATION_GAUSS_KRONROD_POINTS'] = 100
    pyccl.gsl_params['INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS'] = 100
    
    # Multipole
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    
    # Alignment
    with open(os.path.join(info_folder, 'ALIGNMENT.json'), 'r') as file:
        alignment_info = json.load(file)
    alignment_bias = alignment_info['A']
    
    # Cell Average
    cell_average = numpy.zeros((bin_source_size, bin_source_size, ell_size))
    for (i, j) in product(range(bin_source_size), range(bin_source_size)):
        if i <= j:
            tracer1 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, average_source[i, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
            tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, average_source[j, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
            cell_grid = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.01, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
            
            cell_value = scipy.interpolate.CubicSpline(x=numpy.log(ell_grid), y=ell_grid * cell_grid, bc_type='natural', extrapolate=True)
            for k in range(ell_size):
                cell_average[i, j, k] = cell_value.integrate(numpy.log(ell_grid[k]), numpy.log(ell_grid[k + 1])) / (ell_grid[k + 1] - ell_grid[k])
                cell_average[j, i, k] = cell_average[i, j, k]
    
    # Cell Data
    cell_data = numpy.zeros((sample_size, bin_source_size, bin_source_size, ell_size))
    for n in range(sample_size):
        index = sample_index[n]
        print(n)
        for (i, j) in product(range(bin_source_size), range(bin_source_size)):
            if i <= j:
                tracer1 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, data_source[index, i, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
                tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, data_source[index, j, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=grid_size + 1)
                cell_grid = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.01, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
                
                cell_value = scipy.interpolate.CubicSpline(x=numpy.log(ell_grid), y=ell_grid * cell_grid, bc_type='natural', extrapolate=True)
                for k in range(ell_size):
                    cell_data[n, i, j, k] = cell_value.integrate(numpy.log(ell_grid[k]), numpy.log(ell_grid[k + 1])) / (ell_grid[k + 1] - ell_grid[k])
                    cell_data[n, j, i, k] = cell_data[n, i, j, k]
    
    # Save
    with h5py.File(os.path.join(cell_folder, f'{tag}/EE/CORRECT.hdf5'), 'w') as file:
        file.create_dataset('data', data = cell_data)
        file.create_dataset('average', data = cell_average)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Cell EE Correct')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)