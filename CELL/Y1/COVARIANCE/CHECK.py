import argparse
import os
import sys
import time

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import UTILITY


def list_matrix_spots(path, matrix, labels, n_check=8):
    '''
    Compare a few LIST diagonal entries with MATRIX
    
    Arguments:
        path (str): LIST.ascii path
        matrix (numpy.ndarray): Covariance
        labels (list): Constructed labels
        n_check (int): Number of sampled entries
    
    Returns:
        report (dict): Matched examples
    '''
    if not os.path.exists(path):
        return {'exists': False}
    examples = []
    probe_of = {'gggg': 'TT', 'gtgt': 'TE', 'xexe': 'EE', 'mmmm': 'EE'}
    with open(path, 'r') as file:
        file.readline()
        for line in file:
            part = line.split()
            if len(part) < 10 or part[0] not in probe_of:
                continue
            ell1 = float(part[1])
            ell2 = float(part[2])
            if ell1 != ell2:
                continue
            tomoi = int(part[5])
            tomoj = int(part[6])
            tomok = int(part[7])
            tomol = int(part[8])
            if (tomoi, tomoj) != (tomok, tomol):
                continue
            examples.append({
                'probe': probe_of[part[0]],
                'bin_i': tomoi,
                'bin_j': tomoj,
                'ell': ell1,
                'list_value': float(part[9]),
            })
            if len(examples) >= 40:
                break
    _, ell_data = UTILITY.ell_centres()
    mapping = {(item['probe'], item['bin_i'], item['bin_j'], item['ell_bin']): item['index'] for item in labels}
    matched = []
    for example in examples:
        nearest = int(numpy.argmin(numpy.abs(ell_data - example['ell'])))
        key = (example['probe'], example['bin_i'] - 1, example['bin_j'] - 1, nearest)
        if key in mapping:
            index = mapping[key]
            matched.append({
                'probe': example['probe'],
                'bin_i': example['bin_i'],
                'bin_j': example['bin_j'],
                'ell_bin': nearest,
                'ell_list': example['ell'],
                'ell_centre': float(ell_data[nearest]),
                'list_value': example['list_value'],
                'matrix_value': float(matrix[index, index]),
                'rel_diff': float(abs(matrix[index, index] - example['list_value']) / max(abs(example['list_value']), 1e-30)),
            })
        if len(matched) >= n_check:
            break
    return {'exists': True, 'n_matched': len(matched), 'examples': matched}


def inspect_matrix(path, labels, list_path=None):
    '''
    Diagnose one covariance file
    
    Arguments:
        path (str): MATRIX path
        labels (list): Map labels
        list_path (str): Optional LIST path
    
    Returns:
        report (dict): Diagnostics
    '''
    if not os.path.exists(path):
        return {'exists': False, 'path': path}
    matrix = numpy.loadtxt(path)
    report = UTILITY.matrix_diagnostics(matrix, name=os.path.basename(path))
    report['exists'] = True
    report['path'] = path
    same, selected = UTILITY.same_ell_covariance(matrix, labels)
    report['same_ell'] = UTILITY.matrix_diagnostics(same, name='same_ell')
    report['same_ell_blocks'] = UTILITY.block_cholesky_reports(same, selected)
    report['same_ell_blocks_ok'] = bool(report['same_ell_blocks']) and all(item['cholesky_success'] for item in report['same_ell_blocks'])
    if list_path:
        report['list_spots'] = list_matrix_spots(list_path, matrix, labels)
    return report


def main(tag, folder):
    '''
    Check reordered tables and covariance matrices on CELL/{tag}/COVARIANCE
    
    Arguments:
        tag (str): Year tag
        folder (str): Product root
    
    Returns:
        duration (float): Runtime in minutes
    '''
    start = time.time()
    folder = UTILITY.resolve_folder(folder)
    destination = os.path.join(folder, 'CELL', tag, 'COVARIANCE')
    print(f'Tag: {tag}')
    
    meta = UTILITY.load_meta(folder, tag)
    bin_lens_size = int(meta['bin_lens_size'])
    bin_source_size = int(meta['bin_source_size'])
    mapping, labels = UTILITY.build_covariance_map(bin_lens_size, bin_source_size)
    te_keys = [(i, j) for i in range(bin_lens_size) for j in range(bin_source_size)]
    te_indices = [mapping[('TE', i, j, 0)] for i, j in te_keys]
    
    tables = {}
    for name, n1, n2 in [
        ('Cell_gg.ascii', bin_lens_size, bin_lens_size),
        ('Cell_gkappa.ascii', bin_lens_size, bin_source_size),
        ('Cell_kappakappa.ascii', bin_source_size, bin_source_size),
    ]:
        path = os.path.join(destination, name)
        loaded = numpy.loadtxt(path)
        tables[name] = UTILITY.reshape_agrees_with_labels(loaded[:, 0], loaded[:, 1], loaded[:, 2], loaded[:, 3], n1, n2)
    
    matrices = {}
    for name, list_name in [
        ('MATRIX_GAUSS.ascii', 'LIST_GAUSS.ascii'),
        ('MATRIX.ascii', 'LIST.ascii'),
        ('MATRIX_nongauss.ascii', None),
        ('MATRIX_SSC.ascii', None),
    ]:
        list_path = os.path.join(destination, list_name) if list_name else None
        matrices[name] = inspect_matrix(os.path.join(destination, name), labels, list_path)
    
    components = {}
    if all(matrices[name].get('exists') for name in ['MATRIX.ascii', 'MATRIX_GAUSS.ascii', 'MATRIX_nongauss.ascii', 'MATRIX_SSC.ascii']):
        total = numpy.loadtxt(os.path.join(destination, 'MATRIX.ascii'))
        parts = (
            numpy.loadtxt(os.path.join(destination, 'MATRIX_GAUSS.ascii'))
            + numpy.loadtxt(os.path.join(destination, 'MATRIX_nongauss.ascii'))
            + numpy.loadtxt(os.path.join(destination, 'MATRIX_SSC.ascii'))
        )
        components = {
            'max_abs_diff': float(numpy.max(numpy.abs(total - parts))),
            'rel_diff': float(numpy.max(numpy.abs(total - parts)) / max(numpy.max(numpy.abs(total)), 1e-30)),
        }
    
    report = {
        'tag': tag,
        'destination': destination,
        'tables': tables,
        'all_reshape_agree': bool(all(item['reshape_agrees'] for item in tables.values())),
        'te_map_unique_pairs': {
            'n_pairs': len(te_keys),
            'n_unique_indices': len(set(te_indices)),
            'ok': len(te_keys) == len(set(te_indices)),
        },
        'matrices': matrices,
        'sum_of_components': components,
        'regularisation': 'none',
    }
    UTILITY.write_json(os.path.join(destination, 'CHECK.json'), report)
    duration = (time.time() - start) / 60.0
    print(f'Wrote {os.path.join(destination, "CHECK.json")}')
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    PARSE = argparse.ArgumentParser(description='Check covariance')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The product root')
    ARG = PARSE.parse_args()
    OUTPUT = main(ARG.tag, ARG.folder)
