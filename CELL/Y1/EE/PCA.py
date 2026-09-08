import argparse
import json
import os
import time
from itertools import product

import h5py
import numpy
import pyccl
import scipy


def main(tag, folder, sample_size, seed):
    '''
    Calculate the shape-shape angular power spectra of latent-matched PCA n(z)
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the dataset
        sample_size (int): Number of seeded rows
        seed (int): Subset seed
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print(f'Tag: {tag}')
    
    cell_folder = os.path.join(folder, 'CELL/')
    info_folder = os.path.join(folder, 'INFO/')
    source_folder = os.path.join(folder, 'SOURCE/')
    os.makedirs(os.path.join(cell_folder, f'{tag}/'), exist_ok=True)
    os.makedirs(os.path.join(cell_folder, f'{tag}/EE/'), exist_ok=True)
    
    with h5py.File(os.path.join(source_folder, f'{tag}/TRUNCATE.hdf5'), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta']}
        data_source = file['data']['output'][...]
    available = data_source.shape[0]
    if sample_size is None or sample_size >= available:
        index = numpy.arange(available)
    else:
        index = numpy.sort(numpy.random.RandomState(seed).choice(available, size=sample_size, replace=False))
    data_source = data_source[index]
    average_source = numpy.average(data_source, axis=0)
    bin_source_size = int(meta['bin_size'][...])
    sample_size = data_source.shape[0]
    
    z1 = 0.0
    z2 = 3.0
    z_size = int(meta['z_size'][...])
    z_grid = numpy.linspace(z1, z2, z_size)
    
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
    
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    
    with open(os.path.join(info_folder, 'ALIGNMENT.json'), 'r') as file:
        alignment_info = json.load(file)
    alignment_bias = alignment_info['A']
    
    cell_average = numpy.zeros((bin_source_size, bin_source_size, ell_size))
    for (i, j) in product(range(bin_source_size), range(bin_source_size)):
        if i <= j:
            tracer1 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, average_source[i, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=z_size)
            tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, average_source[j, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=z_size)
            cell_grid = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.01, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
            cell_value = scipy.interpolate.CubicSpline(x=numpy.log(ell_grid), y=ell_grid * cell_grid, bc_type='natural', extrapolate=True)
            for k in range(ell_size):
                cell_average[i, j, k] = cell_value.integrate(numpy.log(ell_grid[k]), numpy.log(ell_grid[k + 1])) / (ell_grid[k + 1] - ell_grid[k])
                cell_average[j, i, k] = cell_average[i, j, k]
    
    cell_data = numpy.zeros((sample_size, bin_source_size, bin_source_size, ell_size))
    for n in range(sample_size):
        for (i, j) in product(range(bin_source_size), range(bin_source_size)):
            if i <= j:
                tracer1 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, data_source[n, i, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=z_size)
                tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, data_source[n, j, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=z_size)
                cell_grid = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.01, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
                cell_value = scipy.interpolate.CubicSpline(x=numpy.log(ell_grid), y=ell_grid * cell_grid, bc_type='natural', extrapolate=True)
                for k in range(ell_size):
                    cell_data[n, i, j, k] = cell_value.integrate(numpy.log(ell_grid[k]), numpy.log(ell_grid[k + 1])) / (ell_grid[k + 1] - ell_grid[k])
                    cell_data[n, j, i, k] = cell_data[n, i, j, k]
    
    with h5py.File(os.path.join(cell_folder, f'{tag}/EE/PCA.hdf5'), 'w') as file:
        file.create_dataset('data', data=cell_data)
        file.create_dataset('average', data=cell_average)
        file.create_dataset('index', data=index)
    
    duration = (time.time() - start) / 60
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    PARSE = argparse.ArgumentParser(description='Cell EE PCA')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    PARSE.add_argument('--sample-size', type=int, default=16, help='Number of seeded rows')
    PARSE.add_argument('--seed', type=int, default=1, help='Subset seed')
    ARG = PARSE.parse_args()
    OUTPUT = main(ARG.tag, ARG.folder, ARG.sample_size, ARG.seed)
