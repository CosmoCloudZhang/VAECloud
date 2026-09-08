'''Shared map, cuts, covariance checks, and VALUE histograms.'''

import json
import os
import time
from itertools import product

import h5py
import numpy
import pyccl
import scipy
from matplotlib import pyplot


DEFAULT_FOLDER = '/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/VAECloud'
ELL1 = 20
ELL2 = 2000
ELL_SIZE = 20
K_MAX_DENSITY = 0.1
K_MAX_SHEAR = 10.0


def resolve_folder(folder=None):
    '''
    Resolve the saved-product root from the environment
    
    Arguments:
        folder (str): Optional explicit product root
    
    Returns:
        folder (str): Existing product root
    '''
    candidates = [
        folder,
        os.environ.get('VAECLOUD_FOLDER'),
        DEFAULT_FOLDER,
        os.path.join(os.environ.get('CFS', '/global/cfs/cdirs'), 'lsst/groups/MCP/CosmoCloud/VAECloud'),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, 'DATA')):
            return candidate
    raise FileNotFoundError('Unable to resolve VAECloud product root')


def json_safe(value):
    '''
    Convert numpy objects into JSON-safe Python types
    
    Arguments:
        value (object): Value to convert
    
    Returns:
        value (object): JSON-safe value
    '''
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, (numpy.floating,)):
        return float(value)
    if isinstance(value, (numpy.integer,)):
        return int(value)
    if isinstance(value, (numpy.bool_,)):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path, payload):
    '''
    Write a JSON record
    
    Arguments:
        path (str): Output path
        payload (object): Record
    
    Returns:
        path (str): Output path
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        json.dump(json_safe(payload), file, indent=2, sort_keys=True)
    return path


def unique_pairs(size):
    '''
    Return unique tomographic pairs with i <= j
    
    Arguments:
        size (int): Number of bins
    
    Returns:
        pairs (list): List of (i, j)
    '''
    return [(i, j) for i in range(size) for j in range(i, size)]


def expected_counts(bin_lens_size, bin_source_size, ell_size=ELL_SIZE):
    '''
    Expected pre-cut data-vector lengths
    
    Arguments:
        bin_lens_size (int): Number of lens bins
        bin_source_size (int): Number of source bins
        ell_size (int): Number of multipole bins
    
    Returns:
        counts (dict): Probe lengths and total
    '''
    n_tt = bin_lens_size * (bin_lens_size + 1) // 2
    n_te = bin_lens_size * bin_source_size
    n_ee = bin_source_size * (bin_source_size + 1) // 2
    return {
        'n_tt': n_tt,
        'n_te': n_te,
        'n_ee': n_ee,
        'tt': n_tt * ell_size,
        'te': n_te * ell_size,
        'ee': n_ee * ell_size,
        'total': (n_tt + n_te + n_ee) * ell_size,
    }


def build_covariance_map(bin_lens_size, bin_source_size, ell_size=ELL_SIZE):
    '''
    Build the OneCovariance matrix index map
    
    Arguments:
        bin_lens_size (int): Number of lens bins
        bin_source_size (int): Number of source bins
        ell_size (int): Number of multipole bins
    
    Returns:
        mapping (dict): Keys (probe, i, j, ell_bin) to matrix index
        labels (list): Ordered label dictionaries
    '''
    mapping = {}
    labels = []
    index = 0
    for i, j in unique_pairs(bin_lens_size):
        for ell_bin in range(ell_size):
            key = ('TT', i, j, ell_bin)
            mapping[key] = index
            labels.append({'index': index, 'probe': 'TT', 'bin_i': i, 'bin_j': j, 'ell_bin': ell_bin})
            index += 1
    for i in range(bin_lens_size):
        for j in range(bin_source_size):
            for ell_bin in range(ell_size):
                key = ('TE', i, j, ell_bin)
                mapping[key] = index
                labels.append({'index': index, 'probe': 'TE', 'bin_i': i, 'bin_j': j, 'ell_bin': ell_bin})
                index += 1
    for i, j in unique_pairs(bin_source_size):
        for ell_bin in range(ell_size):
            key = ('EE', i, j, ell_bin)
            mapping[key] = index
            labels.append({'index': index, 'probe': 'EE', 'bin_i': i, 'bin_j': j, 'ell_bin': ell_bin})
            index += 1
    return mapping, labels


def ell_centres(ell1=ELL1, ell2=ELL2, ell_size=ELL_SIZE):
    '''
    Geometric multipole-bin centres used by the CELL scripts
    
    Arguments:
        ell1 (float): Minimum multipole
        ell2 (float): Maximum multipole
        ell_size (int): Number of bins
    
    Returns:
        ell_grid (numpy.ndarray): Bin edges
        ell_data (numpy.ndarray): Geometric centres
    '''
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    ell_data = numpy.sqrt(ell_grid[1:] * ell_grid[:-1])
    return ell_grid, ell_data


def midpoint(edge, index):
    '''
    Midpoint of a tomographic bin
    
    Arguments:
        edge (numpy.ndarray): Bin edges
        index (int): Bin index
    
    Returns:
        value (float): Midpoint redshift
    '''
    return (edge[index] + edge[index + 1]) / 2.0


def scale_cut_ell(cosmology, cosmology_info, edge, k_max_h):
    '''
    Multipole scale cuts from k_max and comoving distance
    
    Arguments:
        cosmology (pyccl.Cosmology): Cosmology
        cosmology_info (dict): Saved cosmology dictionary
        edge (numpy.ndarray): Redshift bin edges
        k_max_h (float): k_max in h Mpc^{-1}
    
    Returns:
        ell_centre (numpy.ndarray): Cuts at bin centres
        ell_edge (numpy.ndarray): Legacy cuts at the supplied edge array
    '''
    k_maximal = k_max_h * cosmology_info['H']
    z_centre = 0.5 * (edge[:-1] + edge[1:])
    ell_centre = k_maximal * pyccl.comoving_radial_distance(cosmo=cosmology, a=1.0 / (1.0 + z_centre)) - 0.5
    ell_edge = k_maximal * pyccl.comoving_radial_distance(cosmo=cosmology, a=1.0 / (1.0 + edge)) - 0.5
    return numpy.array(ell_centre), numpy.array(ell_edge)


def selected_te_pairs(bin_lens, bin_source):
    '''
    Foreground-lens / background-source pairs using bin midpoints
    
    Arguments:
        bin_lens (numpy.ndarray): Lens edges
        bin_source (numpy.ndarray): Source edges
    
    Returns:
        pairs (list): Selected (i, j)
    '''
    pairs = []
    for i in range(len(bin_lens) - 1):
        for j in range(len(bin_source) - 1):
            if midpoint(bin_lens, i) < midpoint(bin_source, j):
                pairs.append((i, j))
    return pairs


def reshape_agrees_with_labels(ell, index1, index2, value, n_tomo_1, n_tomo_2):
    '''
    Test whether a direct reshape matches label-based assembly
    
    Arguments:
        ell (numpy.ndarray): Flattened multipoles
        index1 (numpy.ndarray): Flattened first indices
        index2 (numpy.ndarray): Flattened second indices
        value (numpy.ndarray): Flattened values
        n_tomo_1 (int): First tomographic size
        n_tomo_2 (int): Second tomographic size
    
    Returns:
        report (dict): Agreement diagnostics
    '''
    ell = numpy.asarray(ell)
    index1 = numpy.asarray(index1)
    index2 = numpy.asarray(index2)
    value = numpy.asarray(value)
    unique_ell = numpy.unique(ell)
    n_ell = unique_ell.size
    expected = n_ell * n_tomo_1 * n_tomo_2
    report = {
        'n_row': int(value.size),
        'n_ell': int(n_ell),
        'expected_row': int(expected),
        'n_distinct_orders': None,
        'reshape_agrees': False,
        'max_abs_diff': None,
        'orders': [],
    }
    if value.size != expected:
        report['error'] = f'row count {value.size} != {expected}'
        return report
    assembled = numpy.full((n_ell, n_tomo_1, n_tomo_2), numpy.nan)
    orders = []
    for ell_index, ell_value in enumerate(unique_ell):
        select = numpy.isclose(ell, ell_value)
        pair = list(zip(index1[select].astype(int), index2[select].astype(int)))
        orders.append(tuple(pair))
        for tomo_i, tomo_j, cell in zip(index1[select], index2[select], value[select]):
            assembled[ell_index, int(tomo_i) - 1, int(tomo_j) - 1] = cell
    report['n_distinct_orders'] = int(len(set(orders)))
    try:
        reshaped = value.reshape((n_ell, n_tomo_1, n_tomo_2))
        finite = numpy.isfinite(assembled)
        diff = numpy.max(numpy.abs(reshaped[finite] - assembled[finite])) if finite.any() else numpy.nan
        report['max_abs_diff'] = float(diff)
        report['reshape_agrees'] = bool(finite.all() and diff == 0.0)
    except Exception as error:
        report['error'] = str(error)
    return report


def matrix_diagnostics(matrix, name='matrix'):
    '''
    Report covariance usability without regularisation
    
    Arguments:
        matrix (numpy.ndarray): Covariance
        name (str): Label
    
    Returns:
        report (dict): Shape, symmetry, eigenvalues and Cholesky
    '''
    matrix = numpy.asarray(matrix, dtype=numpy.float64)
    report = {
        'name': name,
        'shape': list(matrix.shape),
        'finite': bool(numpy.isfinite(matrix).all()),
        'positive_diagonal': None,
        'min_diagonal': None,
        'max_diagonal': None,
        'scaled_symmetry_error': None,
        'min_eigenvalue': None,
        'max_eigenvalue': None,
        'max_abs_correlation': None,
        'cholesky_success': False,
        'cholesky_error': None,
    }
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        report['cholesky_error'] = 'not square'
        return report
    diagonal = numpy.diag(matrix)
    report['min_diagonal'] = float(numpy.min(diagonal)) if diagonal.size else None
    report['max_diagonal'] = float(numpy.max(diagonal)) if diagonal.size else None
    report['positive_diagonal'] = bool(numpy.all(diagonal > 0.0)) if diagonal.size else False
    scale = numpy.maximum(numpy.abs(matrix), 1.0)
    report['scaled_symmetry_error'] = float(numpy.max(numpy.abs(matrix - matrix.T) / scale))
    try:
        eigenvalues = numpy.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        report['min_eigenvalue'] = float(numpy.min(eigenvalues))
        report['max_eigenvalue'] = float(numpy.max(eigenvalues))
    except Exception as error:
        report['cholesky_error'] = str(error)
        return report
    if report['positive_diagonal']:
        correlation = matrix / numpy.sqrt(numpy.outer(diagonal, diagonal))
        off = correlation - numpy.eye(matrix.shape[0])
        report['max_abs_correlation'] = float(numpy.max(numpy.abs(off)))
    try:
        numpy.linalg.cholesky(matrix)
        report['cholesky_success'] = True
    except numpy.linalg.LinAlgError as error:
        report['cholesky_error'] = str(error)
    return report


def same_ell_covariance(matrix, labels, mask=None):
    '''
    Keep same-multipole correlations and zero the rest
    
    Arguments:
        matrix (numpy.ndarray): Full covariance
        labels (list): Ordered map labels
        mask (numpy.ndarray): Optional retained-row mask
    
    Returns:
        same (numpy.ndarray): Same-ell covariance on the selected rows
        selected_labels (list): Retained labels
    '''
    matrix = numpy.asarray(matrix, dtype=numpy.float64)
    if mask is None:
        mask = numpy.ones(matrix.shape[0], dtype=bool)
    selected = numpy.where(mask)[0]
    selected_labels = [labels[index] for index in selected]
    same = matrix[numpy.ix_(selected, selected)].copy()
    ell = numpy.array([item['ell_bin'] for item in selected_labels])
    same[ell[:, None] != ell[None, :]] = 0.0
    return same, selected_labels


def block_cholesky_reports(matrix, labels):
    '''
    Cholesky each same-multipole block
    
    Arguments:
        matrix (numpy.ndarray): Same-ell covariance
        labels (list): Labels of the retained rows
    
    Returns:
        reports (list): Per-ell diagnostics
    '''
    reports = []
    ell_bins = sorted(set(item['ell_bin'] for item in labels))
    for ell_bin in ell_bins:
        index = [n for n, item in enumerate(labels) if item['ell_bin'] == ell_bin]
        block = numpy.asarray(matrix[numpy.ix_(index, index)], dtype=numpy.float64)
        reports.append(matrix_diagnostics(block, name=f'ell_{ell_bin}'))
    return reports


def weighted_q(delta, matrix, mode='full'):
    '''
    Covariance-weighted displacement per data entry
    
    Arguments:
        delta (numpy.ndarray): Residuals, shape (n_object, n_data)
        matrix (numpy.ndarray): Covariance
        mode (str): Approximation name
    
    Returns:
        value (numpy.ndarray): chi^2 / N_D for each object
        status (dict): Factorisation status
    '''
    delta = numpy.asarray(delta, dtype=numpy.float64)
    matrix = numpy.asarray(matrix, dtype=numpy.float64)
    status = {'mode': mode, 'n_data': int(delta.shape[-1]), 'cholesky_success': False}
    if delta.size == 0:
        status['error'] = 'empty residual'
        return numpy.full(delta.shape[0], numpy.nan), status
    try:
        factor = numpy.linalg.cholesky(matrix)
        solve = numpy.linalg.solve(factor, delta.T)
        quadratic = numpy.sum(solve * solve, axis=0)
        status['cholesky_success'] = True
        return quadratic / delta.shape[-1], status
    except numpy.linalg.LinAlgError as error:
        status['error'] = str(error)
        return numpy.full(delta.shape[0], numpy.nan), status


def diagonal_q(delta, variance):
    '''
    Diagonal covariance-weighted displacement per data entry
    
    Arguments:
        delta (numpy.ndarray): Residuals
        variance (numpy.ndarray): Diagonal variances
    
    Returns:
        value (numpy.ndarray): Q_diag
    '''
    delta = numpy.asarray(delta, dtype=numpy.float64)
    variance = numpy.asarray(variance, dtype=numpy.float64)
    return numpy.sum(numpy.square(delta) / variance, axis=1) / delta.shape[-1]


def load_meta(folder, tag):
    '''
    Load DATA/{tag}/CORRECT.hdf5 metadata
    
    Arguments:
        folder (str): Product root
        tag (str): Year tag
    
    Returns:
        meta (dict): Arrays from the meta group
    '''
    with h5py.File(os.path.join(folder, 'DATA', tag, 'CORRECT.hdf5'), 'r') as file:
        return {key: file['meta'][key][...] for key in file['meta']}


def configure_pyplot():
    '''
    Apply the existing thesis plot style
    
    Arguments:
        None
    
    Returns:
        None
    '''
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ.get('PATH', '')
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20


def load_cube(path, index=None):
    '''
    Load a CELL cube
    
    Arguments:
        path (str): HDF5 path
        index (numpy.ndarray): Optional row index
    
    Returns:
        data (numpy.ndarray): Spectra
        stored (numpy.ndarray): Stored row index or None
    '''
    with h5py.File(path, 'r') as file:
        data = file['data'][...]
        stored = file['index'][...] if 'index' in file else None
    if index is not None:
        data = data[index]
    return data, stored


def cosmology_from_info(info_folder):
    '''
    Build the project CCL cosmology
    
    Arguments:
        info_folder (str): INFO directory
    
    Returns:
        cosmology (pyccl.Cosmology): Cosmology
        cosmology_info (dict): Saved dictionary
    '''
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
    return cosmology, cosmology_info


def integrate_cell(cosmology, tracer1, tracer2, ell_grid):
    '''
    Band-average C_ell on the CELL grid
    
    Arguments:
        cosmology (pyccl.Cosmology): Cosmology
        tracer1 (object): First tracer
        tracer2 (object): Second tracer
        ell_grid (numpy.ndarray): Multipole edges
    
    Returns:
        value (numpy.ndarray): Band-averaged C_ell
    '''
    cell_grid = pyccl.cells.angular_cl(
        cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid,
        p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.01,
        limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False
    )
    spline = scipy.interpolate.CubicSpline(x=numpy.log(ell_grid), y=ell_grid * cell_grid, bc_type='natural', extrapolate=True)
    value = numpy.zeros(ell_grid.size - 1)
    for k in range(value.size):
        value[k] = spline.integrate(numpy.log(ell_grid[k]), numpy.log(ell_grid[k + 1])) / (ell_grid[k + 1] - ell_grid[k])
    return value


def reference_spectra(folder, tag, cosmology):
    '''
    C_ell of the common mean n(z) from VAE data/input
    
    Arguments:
        folder (str): Product root
        tag (str): Year tag
        cosmology (pyccl.Cosmology): Cosmology
    
    Returns:
        cubes (dict): TT, TE, EE reference spectra
    '''
    info_folder = os.path.join(folder, 'INFO')
    with h5py.File(os.path.join(folder, 'LENS', tag, 'VAE.hdf5'), 'r') as file:
        z_size = int(file['meta']['z_size'][...])
        mean_lens = numpy.mean(file['data']['input'][...], axis=0)
        bin_lens_size = int(file['meta']['bin_size'][...])
    with h5py.File(os.path.join(folder, 'SOURCE', tag, 'VAE.hdf5'), 'r') as file:
        mean_source = numpy.mean(file['data']['input'][...], axis=0)
        bin_source_size = int(file['meta']['bin_size'][...])
    z_grid = numpy.linspace(0.0, 3.0, z_size)
    ell_grid, _ = ell_centres()
    with open(os.path.join(info_folder, 'GALAXY.json'), 'r') as file:
        galaxy_bias = json.load(file)[tag]
    with open(os.path.join(info_folder, 'MAGNIFICATION.json'), 'r') as file:
        magnification_bias = json.load(file)[tag]
    with open(os.path.join(info_folder, 'ALIGNMENT.json'), 'r') as file:
        alignment_bias = json.load(file)['A']
    
    tt = numpy.zeros((bin_lens_size, bin_lens_size, ELL_SIZE))
    for i, j in product(range(bin_lens_size), range(bin_lens_size)):
        if i <= j:
            tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, mean_lens[i, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, numpy.ones(z_size) * magnification_bias[i]], has_rsd=False, n_samples=z_size)
            tracer2 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, mean_lens[j, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, numpy.ones(z_size) * magnification_bias[j]], has_rsd=False, n_samples=z_size)
            tt[i, j, :] = integrate_cell(cosmology, tracer1, tracer2, ell_grid)
            tt[j, i, :] = tt[i, j, :]
    te = numpy.zeros((bin_lens_size, bin_source_size, ELL_SIZE))
    for i, j in product(range(bin_lens_size), range(bin_source_size)):
        tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, mean_lens[i, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, numpy.ones(z_size) * magnification_bias[i]], has_rsd=False, n_samples=z_size)
        tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, mean_source[j, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=z_size)
        te[i, j, :] = integrate_cell(cosmology, tracer1, tracer2, ell_grid)
    ee = numpy.zeros((bin_source_size, bin_source_size, ELL_SIZE))
    for i, j in product(range(bin_source_size), range(bin_source_size)):
        if i <= j:
            tracer1 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, mean_source[i, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=z_size)
            tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=[z_grid, mean_source[j, :]], has_shear=True, ia_bias=[z_grid, alignment_bias], use_A_ia=False, n_samples=z_size)
            ee[i, j, :] = integrate_cell(cosmology, tracer1, tracer2, ell_grid)
            ee[j, i, :] = ee[i, j, :]
    return {'TT': tt, 'TE': te, 'EE': ee}


def assemble(mapping, labels, cubes, mask_kind, bin_lens, bin_source, cosmology, cosmology_info):
    '''
    Assemble a data vector from the explicit covariance map
    
    Arguments:
        mapping (dict): Index map
        labels (list): Ordered labels
        cubes (dict): TT/TE/EE cubes, shape (n, bin, bin, ell) or (bin, bin, ell)
        mask_kind (str): SINGLE, DOUBLE or TRIPLE
        bin_lens (numpy.ndarray): Lens edges
        bin_source (numpy.ndarray): Source edges
        cosmology (object): CCL cosmology
        cosmology_info (dict): Cosmology dictionary
    
    Returns:
        vector (numpy.ndarray): Data vectors
        mask (numpy.ndarray): Retained entries
        cut_record (dict): Scale-cut record
    '''
    sample = None
    for probe in ['TT', 'TE', 'EE']:
        if cubes.get(probe) is not None:
            sample = cubes[probe]
            break
    n_object = 1 if sample.ndim == 3 else sample.shape[0]
    vector = numpy.zeros((n_object, len(labels)))
    mask = numpy.zeros(len(labels), dtype=bool)
    _, ell_data = ell_centres()
    ell_tt, ell_tt_edge = scale_cut_ell(cosmology, cosmology_info, bin_lens, K_MAX_DENSITY)
    ell_ee, ell_ee_edge = scale_cut_ell(cosmology, cosmology_info, bin_source, K_MAX_SHEAR)
    te_pairs = set(selected_te_pairs(bin_lens, bin_source))
    for label in labels:
        probe, i, j, ell_bin = label['probe'], label['bin_i'], label['bin_j'], label['ell_bin']
        index = label['index']
        cube = cubes.get(probe)
        if cube is None:
            continue
        value = cube[i, j, ell_bin] if cube.ndim == 3 else cube[:, i, j, ell_bin]
        if probe == 'TT' and mask_kind in ('DOUBLE', 'TRIPLE'):
            vector[:, index] = value
            mask[index] = ell_data[ell_bin] < min(ell_tt[i], ell_tt[j])
        elif probe == 'TE' and mask_kind in ('DOUBLE', 'TRIPLE'):
            vector[:, index] = value
            mask[index] = ((i, j) in te_pairs) and (ell_data[ell_bin] < ell_tt[i])
        elif probe == 'EE' and mask_kind in ('SINGLE', 'TRIPLE'):
            vector[:, index] = value
            mask[index] = ell_data[ell_bin] < min(ell_ee[i], ell_ee[j])
    return vector, mask, {'ell_tt_centre': ell_tt, 'ell_ee_centre': ell_ee, 'ell_tt_edge': ell_tt_edge, 'ell_ee_edge': ell_ee_edge, 'te_pairs': list(te_pairs)}


def comma_list(values):
    '''
    Join values as a comma-separated OneCovariance list
    
    Arguments:
        values (iterable): Numbers or strings
    
    Returns:
        text (str): Comma-separated list
    '''
    return ', '.join(str(value) for value in values)


def write_onecovariance_ini(path, destination, kind, cosmology, n_eff_clust, n_eff_lensing, survey_area, n_source, sigma8=0.808686125499388):
    '''
    Write a Gaussian or full OneCovariance ini
    
    Arguments:
        path (str): Output ini path
        destination (str): CELL/{tag}/COVARIANCE directory
        kind (str): gaussian or full
        cosmology (dict): INFO/COSMOLOGY.json
        n_eff_clust (list): Lens n_eff per bin
        n_eff_lensing (list): Source n_eff per bin
        survey_area (float): Survey area in deg^2
        n_source (int): Number of source bins
        sigma8 (float): Fiducial sigma8
    
    Returns:
        None
    '''
    kind = kind.lower()
    if kind in ('gauss', 'gaussian'):
        split_gauss, nongauss, ssc = 'False', 'False', 'False'
        file_names = 'LIST_GAUSS.ascii, MATRIX_GAUSS.ascii'
        corr_name = 'CORRELATION_GAUSS.pdf'
        save_name = 'VALUE_GAUSS.ini'
    elif kind == 'full':
        split_gauss, nongauss, ssc = 'True', 'True', 'True'
        file_names = 'LIST.ascii, MATRIX.ascii'
        corr_name = 'CORRELATION.pdf'
        save_name = 'VALUE.ini'
    else:
        raise ValueError(f'kind must be gaussian or full, got {kind}')
    zeros = comma_list([0] * n_source)
    ellipticity = comma_list([0.26] * n_source)
    text = f'''; OneCovariance configuration written by CONFIG.py ({kind})

[covariance terms]
gauss = True
split_gauss = {split_gauss}
nongauss = {nongauss}
ssc = {ssc}

[observables]
cosmic_shear = True
est_shear = C_ell
ggl = True
est_ggl = C_ell
clustering = True
est_clust = C_ell
cstellar_mf = False
cross_terms = True
unbiased_clustering = False

[output settings]
directory = {destination}
file = {file_names}
style = list, matrix
list_style_spatial_first = True
corrmatrix_plot = {corr_name}
save_configs = {save_name}
save_Cells = False
save_trispectra = False
save_alms = True
use_tex = True

[covELLspace settings]
delta_z = 0.01
tri_delta_z = 0.10
integration_steps = 500
nz_interpolation_polynom_order = 1
mult_shear_bias = {zeros}

ell_min = 20
ell_max = 2000
ell_bins = 100
ell_type = log

ell_min_clustering = 20
ell_max_clustering = 2000
ell_bins_clustering = 20
ell_type_clustering = log

ell_min_lensing = 20
ell_max_lensing = 2000
ell_bins_lensing = 20
ell_type_lensing = log

[covTHETAspace settings]
theta_min = 0.5
theta_max = 300.0
theta_bins = 9
theta_type = log

theta_list = 1, 2, 3
xi_pp = True
xi_mm = True
theta_accuracy = 1e-3
integration_intervals = 400

[survey specs]
survey_area_clust_in_deg2 = {float(survey_area)}
survey_area_ggl_in_deg2 = {float(survey_area)}
survey_area_lensing_in_deg2 = {float(survey_area)}
ellipticity_dispersion = {ellipticity}
n_eff_clust = {comma_list(n_eff_clust)}
n_eff_lensing = {comma_list(n_eff_lensing)}

[redshift]
z_directory = {destination}
zclust_file = LENS.ascii
value_loc_in_clustbin = left
zlens_file = SOURCE.ascii
value_loc_in_lensbin = left

[cosmo]
w0 = {cosmology['W0']}
wa = {cosmology['WA']}
h = {cosmology['H']}
ns = {cosmology['NS']}
m_nu = {cosmology['M_NU']}
neff = {cosmology['N_EFF']}
tcmb0 = 2.7255
sigma8 = {sigma8}
omega_de = {cosmology['OMEGA_DE']}
omega_m = {cosmology['OMEGA_M']}
omega_b = {cosmology['OMEGA_B']}

[bias]
bias_files = {os.path.join(destination, 'GALAXY.ascii')}

[IA]
A_IA = 0.5
eta_IA = 0.0
z_pivot_IA = 0.5

[hod]
model_mor_cen = double_powerlaw
model_mor_sat = double_powerlaw
dpow_logm0_cen = 10.51
dpow_logm1_cen = 11.38
dpow_a_cen = 7.096
dpow_b_cen = 0.2
dpow_norm_cen = 1.0
dpow_norm_sat = 0.56
model_scatter_cen = lognormal
model_scatter_sat = modschechter
logn_sigma_c_cen = 0.35
modsch_logmref_sat = 13.0
modsch_alpha_s_sat = -0.858
modsch_b_sat = -0.024, 1.149

[halomodel evaluation]
m_bins = 900
log10m_min = 6
log10m_max = 18
hmf_model = Tinker10
mdef_model = SOMean
mdef_params = overdensity, 200
disable_mass_conversion = True
delta_c = 1.686
transfer_model = CAMB
small_k_damping_for1h = damped

[powspec evaluation]
non_linear_model = mead2020
HMCode_logT_AGN = 7.8
log10k_bins = 500
log10k_min = -4.0
log10k_max = +2.5

[trispec evaluation]
log10k_bins = 500
log10k_min = -4.0
log10k_max = +2.5
matter_klim = 0.001
matter_mulim = 0.001
lower_calc_limit = 1e-200
small_k_damping_for1h = damped

[tabulated inputs files]
Cell_directory = {destination}
Cgg_file = Cell_gg.ascii
Cgm_file = Cell_gkappa.ascii
Cmm_file = Cell_kappakappa.ascii

[misc]
num_cores = 128
'''
    with open(path, 'w') as file:
        file.write(text)


def choose_matrix(cov_dir, covariance='gauss'):
    '''
    Select the Gaussian or full covariance file
    
    Arguments:
        cov_dir (str): CELL/{tag}/COVARIANCE directory
        covariance (str): gauss or full
    
    Returns:
        path (str): Matrix path
        name (str): Approximation name
    '''
    covariance = covariance.lower()
    if covariance in ('gauss', 'gaussian'):
        path = os.path.join(cov_dir, 'MATRIX_GAUSS.ascii')
        name = 'gaussian'
    elif covariance == 'full':
        path = os.path.join(cov_dir, 'MATRIX.ascii')
        name = 'full'
    else:
        raise ValueError(f'covariance must be gauss or full, got {covariance}')
    if not os.path.exists(path):
        raise FileNotFoundError(f'No {name} covariance at {path}')
    return path, name


def plot_chi(path, values, title):
    '''
    Plot chi^2 / N_D histograms and ECDFs
    
    Arguments:
        path (str): Output PDF
        values (dict): Arrays keyed by display name
        title (str): Axis label
    
    Returns:
        None
    '''
    configure_pyplot()
    colors = {
        'PCA': 'darkgreen',
        'VAE': 'darkred',
        'SHIFT': 'darkblue',
        'SCALE': 'darkorange',
        'CORRECT': 'black',
    }
    figure, plot = pyplot.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
    finite = [values[name][numpy.isfinite(values[name])] for name in values]
    finite = [item for item in finite if item.size]
    if not finite:
        pyplot.close(figure)
        return
    x1 = 0.0
    x2 = 1.5
    bins = numpy.linspace(x1, x2, 40)
    for name, sample in values.items():
        sample = sample[numpy.isfinite(sample)]
        if sample.size == 0:
            continue
        plot[0].hist(sample, bins=bins, color=colors.get(name, 'grey'), histtype='step', linewidth=2.5, density=True, label=name, rasterized=True)
        order = numpy.sort(sample)
        plot[1].plot(order, numpy.linspace(0.0, 1.0, order.size), color=colors.get(name, 'grey'), linewidth=2.5, rasterized=True)
    plot[0].legend(fontsize=16, frameon=True, loc='upper right')
    plot[0].set_ylabel(r'$\mathrm{Density}$')
    plot[1].set_ylabel(r'$\mathrm{ECDF}$')
    plot[1].set_xlabel(title)
    plot[1].set_xlim(x1, x2)
    plot[1].set_ylim(0.0, 1.0)
    figure.subplots_adjust(hspace=0.05)
    figure.savefig(path, bbox_inches='tight', dpi=512)
    pyplot.close(figure)


def main(tag, folder, mask_kind='TRIPLE', covariance='full', sample_size=1024, seed=1):
    '''
    Covariance-weighted displacement histograms
    
    Arguments:
        tag (str): Year tag
        folder (str): Product root
        mask_kind (str): SINGLE, DOUBLE or TRIPLE
        covariance (str): gauss or full
        sample_size (int): Comparison size
        seed (int): Subset seed
    
    Returns:
        duration (float): Runtime in minutes
    '''
    start = time.time()
    folder = resolve_folder(folder)
    mask_kind = mask_kind.upper()
    output = os.path.join(folder, 'VALUE', tag)
    os.makedirs(output, exist_ok=True)
    print(f'Tag: {tag}')
    print(f'Mask: {mask_kind}')
    print(f'Covariance: {covariance}')
    
    meta = load_meta(folder, tag)
    bin_lens = numpy.array(meta['bin_lens'])
    bin_source = numpy.array(meta['bin_source'])
    bin_lens_size = int(meta['bin_lens_size'])
    bin_source_size = int(meta['bin_source_size'])
    mapping, labels = build_covariance_map(bin_lens_size, bin_source_size)
    cosmology, cosmology_info = cosmology_from_info(os.path.join(folder, 'INFO'))
    
    pca_cell = {}
    pca_index = None
    pca_keep = {}
    for probe in ['TT', 'TE', 'EE']:
        path = os.path.join(folder, 'CELL', tag, probe, 'PCA.hdf5')
        if os.path.exists(path):
            data, stored = load_cube(path)
            pca_cell[probe] = data
            if stored is not None:
                pca_index = numpy.array(stored)
    for sample_name in ['LENS', 'SOURCE']:
        path = os.path.join(folder, sample_name, tag, 'TRUNCATE.hdf5')
        if os.path.exists(path):
            with h5py.File(path, 'r') as file:
                if 'pca_keep' in file['meta']:
                    pca_keep[sample_name] = int(file['meta']['pca_keep'][...])
                else:
                    pca_keep[sample_name] = int(file['meta']['latent_size'][...])
    
    with h5py.File(os.path.join(folder, 'CELL', tag, 'EE', 'VAE.hdf5'), 'r') as file:
        n_vae = file['data'].shape[0]
    if pca_index is not None:
        index = pca_index
    else:
        index = numpy.sort(numpy.random.RandomState(seed).choice(n_vae, size=min(sample_size, n_vae), replace=False))
    
    methods = {}
    pairing = {}
    if pca_cell:
        methods['PCA'] = pca_cell
        pairing['PCA'] = 'paired_vae_test_rows'
    methods['VAE'] = {}
    for probe in ['TT', 'TE', 'EE']:
        data, _ = load_cube(os.path.join(folder, 'CELL', tag, probe, 'VAE.hdf5'), index=index)
        methods['VAE'][probe] = data
    pairing['VAE'] = 'vae_test_row_order'
    
    unpaired = numpy.sort(numpy.random.RandomState(seed + 17).choice(int(meta['data_size']), size=index.size, replace=False))
    for name in ['SHIFT', 'SCALE', 'CORRECT']:
        methods[name] = {}
        for probe in ['TT', 'TE', 'EE']:
            data, _ = load_cube(os.path.join(folder, 'CELL', tag, probe, f'{name}.hdf5'), index=unpaired)
            methods[name][probe] = data
        pairing[name] = 'unpaired_500k_ensemble'
    
    reference = reference_spectra(folder, tag, cosmology)
    centre, mask, cut_record = assemble(mapping, labels, reference, mask_kind, bin_lens, bin_source, cosmology, cosmology_info)
    centre = centre[0]
    
    assembled = {}
    for name, cubes in methods.items():
        vector, mask, _ = assemble(mapping, labels, cubes, mask_kind, bin_lens, bin_source, cosmology, cosmology_info)
        assembled[name] = vector
    
    cov_dir = os.path.join(folder, 'CELL', tag, 'COVARIANCE')
    matrix_path, matrix_name = choose_matrix(cov_dir, covariance)
    matrix = numpy.loadtxt(matrix_path)
    variance = numpy.diag(matrix)
    selected = matrix[numpy.ix_(numpy.where(mask)[0], numpy.where(mask)[0])]
    selected_report = matrix_diagnostics(selected, name=f'masked_{matrix_name}')
    same, same_labels = same_ell_covariance(matrix, labels, mask=mask)
    same_blocks = block_cholesky_reports(same, same_labels)
    if not selected_report['cholesky_success']:
        raise RuntimeError(f'{matrix_name} covariance Cholesky failed; keep --covariance=gauss if the full matrix is not PD')
    
    chi = {}
    chi_diag = {}
    status = {}
    for name, vector in assembled.items():
        delta = vector[:, mask] - centre[mask]
        chi[name], status[name] = weighted_q(delta, selected, mode=matrix_name)
        chi_diag[name] = diagonal_q(delta, variance[mask])
    
    display = {
        'PCA': 'PCA',
        'VAE': 'VAE',
        'SHIFT': 'SHIFT',
        'SCALE': 'SCALE',
        'CORRECT': 'CORRECT',
    }
    plot_chi(
        os.path.join(output, f'{mask_kind}.pdf'),
        {display[name]: chi[name] for name in chi},
        r'$\chi^2 / N_{\mathrm{D}}$'
    )
    
    hdf5_path = os.path.join(output, f'{mask_kind}.hdf5')
    with h5py.File(hdf5_path, 'w') as file:
        file.create_dataset('index_vae', data=index)
        file.create_dataset('index_unpaired', data=unpaired)
        file.create_dataset('mask', data=mask)
        file.create_dataset('centre', data=centre)
        for name, value in chi.items():
            file.create_dataset(f'chi_{name}', data=value)
        for name, value in chi_diag.items():
            file.create_dataset(f'chi_diag_{name}', data=value)
    
    te_keys = [(i, j) for i in range(bin_lens_size) for j in range(bin_source_size)]
    te_indices = [mapping[('TE', i, j, 0)] for i, j in te_keys]
    
    def summarise(items):
        return {
            name: {
                'mean': float(numpy.nanmean(value)),
                'median': float(numpy.nanmedian(value)),
                'min': float(numpy.nanmin(value)),
                'max': float(numpy.nanmax(value)),
                'fraction_above_5': float(numpy.mean(value > 5.0)),
            }
            for name, value in items.items()
        }
    
    report = {
        'tag': tag,
        'mask': mask_kind,
        'n_map': len(labels),
        'n_data': int(numpy.sum(mask)),
        'n_object': int(index.size),
        'seed': int(seed),
        'centre': 'D(mean(phi_reference)) from VAE data/input mean n(z)',
        'cut': {
            'k_max_density': '0.1 h Mpc^{-1}',
            'k_max_shear': '10 h Mpc^{-1}',
            'distance': 'bin centre',
            'legacy_distance': 'bin edge, recorded only',
            'te': 'lens cut; midpoint lens < midpoint source',
        },
        'te_map_unique_pairs': {
            'n_pairs': len(te_keys),
            'n_unique_indices': len(set(te_indices)),
            'ok': len(te_keys) == len(set(te_indices)),
        },
        'pca_keep': pca_keep,
        'pairing': pairing,
        'covariance': {
            'path': matrix_path,
            'approximation': matrix_name,
            'masked': selected_report,
            'same_ell_blocks_ok': bool(same_blocks) and all(item['cholesky_success'] for item in same_blocks),
            'same_ell_blocks': same_blocks,
            'cholesky_success': bool(selected_report['cholesky_success']),
        },
        'chi_status': status,
        'summary': summarise(chi),
        'diagonal_diagnostic': summarise(chi_diag),
        'legacy_edge_cut': json_safe(cut_record),
    }
    write_json(os.path.join(output, f'{mask_kind}.json'), report)
    
    duration = (time.time() - start) / 60.0
    print(f'Wrote {hdf5_path}')
    print(f'Time: {duration:.2f} minutes')
    return duration
