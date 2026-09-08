import argparse
import os
import sys
import time

import numpy
from astropy import table


HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import UTILITY


PROBE_FILE = {
    'TT': ('Cell_gg.ascii', 'Cell_gg'),
    'TE': ('Cell_gkappa.ascii', 'Cell_gkappa'),
    'EE': ('Cell_kappakappa.ascii', 'Cell_kappakappa'),
}


def reorder_table(path, value_name):
    '''
    Reorder an existing Cell table into lexicographic OneCovariance order
    
    Arguments:
        path (str): ASCII table path
        value_name (str): Value column name
    
    Returns:
        report (dict): Before/after reshape tests
    '''
    loaded = table.Table.read(path, format='ascii.commented_header')
    ell = numpy.array(loaded['ell'])
    index1 = numpy.array(loaded['tomo_i'])
    index2 = numpy.array(loaded['tomo_j'])
    value = numpy.array(loaded[value_name])
    n_tomo_1 = int(index1.max())
    n_tomo_2 = int(index2.max())
    before = UTILITY.reshape_agrees_with_labels(ell, index1, index2, value, n_tomo_1, n_tomo_2)
    order = numpy.lexsort((index2, index1, ell))
    ordered = table.Table()
    ordered['ell'] = ell[order]
    ordered['tomo_i'] = index1[order]
    ordered['tomo_j'] = index2[order]
    ordered[value_name] = value[order]
    ordered.write(path, overwrite=True, format='ascii.commented_header')
    after = UTILITY.reshape_agrees_with_labels(ordered['ell'], ordered['tomo_i'], ordered['tomo_j'], ordered[value_name], n_tomo_1, n_tomo_2)
    return {
        'path': path,
        'before': before,
        'after': after,
        'permutation_changed': bool(not numpy.array_equal(order, numpy.arange(order.size))),
    }


def main(tag, folder):
    '''
    Lexsort Cell tables in place on CELL/{tag}/COVARIANCE
    
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
    
    tables = {}
    for probe, (name, value_name) in PROBE_FILE.items():
        tables[probe] = reorder_table(os.path.join(destination, name), value_name)
    
    meta = UTILITY.load_meta(folder, tag)
    bin_lens_size = int(meta['bin_lens_size'])
    bin_source_size = int(meta['bin_source_size'])
    mapping, labels = UTILITY.build_covariance_map(bin_lens_size, bin_source_size)
    te_keys = [(i, j) for i in range(bin_lens_size) for j in range(bin_source_size)]
    te_indices = [mapping[('TE', i, j, 0)] for i, j in te_keys]
    report = {
        'tag': tag,
        'destination': destination,
        'tables': tables,
        'counts': UTILITY.expected_counts(bin_lens_size, bin_source_size),
        'n_map': len(labels),
        'te_map_unique_pairs': {
            'n_pairs': len(te_keys),
            'n_unique_indices': len(set(te_indices)),
            'ok': len(te_keys) == len(set(te_indices)),
        },
        'selected_te_pairs': UTILITY.selected_te_pairs(numpy.array(meta['bin_lens']), numpy.array(meta['bin_source'])),
        'all_reshape_agree': bool(all(item['after']['reshape_agrees'] for item in tables.values())),
    }
    UTILITY.write_json(os.path.join(destination, 'REORDER.json'), report)
    duration = (time.time() - start) / 60.0
    print(f'Wrote {os.path.join(destination, "REORDER.json")}')
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    PARSE = argparse.ArgumentParser(description='Reorder Cell tables')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The product root')
    ARG = PARSE.parse_args()
    OUTPUT = main(ARG.tag, ARG.folder)
