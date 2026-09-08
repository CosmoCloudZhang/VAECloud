import argparse
import json
import os
import pickle
import time

import h5py
import numpy


def main(tag, name, folder):
    '''
    Reconstruct test n(z) with a latent-matched truncation of the saved PCA
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): LENS or SOURCE
        folder (str): The product root
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    name = name.upper()
    output = os.path.join(folder, name, tag)
    os.makedirs(output, exist_ok=True)
    print(f'Tag: {tag}')
    print(f'Name: {name}')
    
    vae_path = os.path.join(folder, name, tag, 'VAE.hdf5')
    pca_path = os.path.join(folder, name, tag, 'PCA.pkl')
    with h5py.File(vae_path, 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta']}
        input_data = file['data']['input'][...]
    latent_size = int(meta['latent_size'])
    data_size, bin_size, z_size = input_data.shape
    
    with open(pca_path, 'rb') as file:
        model = pickle.load(file)
    flat = numpy.reshape(input_data, (data_size, bin_size * z_size))
    score = model.transform(flat)
    truncated = numpy.zeros_like(score)
    truncated[:, :latent_size] = score[:, :latent_size]
    output_data = numpy.reshape(model.inverse_transform(truncated), (data_size, bin_size, z_size))
    
    path = os.path.join(output, 'TRUNCATE.hdf5')
    with h5py.File(path, 'w') as file:
        group = file.create_group('meta')
        for key, value in meta.items():
            group.create_dataset(key, data=value)
        group.create_dataset('pca_keep', data=latent_size)
        data = file.create_group('data')
        data.create_dataset('input', data=input_data)
        data.create_dataset('output', data=output_data)
        data.create_dataset('coefficient', data=truncated)
        data.create_dataset('coefficient_full', data=score)
    
    full = numpy.reshape(model.inverse_transform(score), (data_size, bin_size, z_size))
    report = {
        'tag': tag,
        'name': name,
        'path': path,
        'latent_size': latent_size,
        'pca_components': int(score.shape[1]),
        'shape': list(output_data.shape),
        'max_abs_input_minus_full': float(numpy.max(numpy.abs(full - input_data))),
        'max_abs_truncation': float(numpy.max(numpy.abs(output_data - input_data))),
    }
    with open(os.path.join(output, 'TRUNCATE.json'), 'w') as file:
        json.dump(report, file, indent=2, sort_keys=True)
    
    duration = (time.time() - start) / 60.0
    print(f'Wrote {path}')
    print(f'L={latent_size} of {score.shape[1]} PCs')
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    PARSE = argparse.ArgumentParser(description='Latent-matched PCA')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='LENS or SOURCE')
    PARSE.add_argument('--folder', type=str, required=True, help='The product root')
    ARG = PARSE.parse_args()
    OUTPUT = main(ARG.tag, ARG.name, ARG.folder)
