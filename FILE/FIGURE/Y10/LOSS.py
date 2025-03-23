import os
import time
import h5py
import numpy
import argparse
from matplotlib import pyplot


def main(tag, folder):
    '''
    Plot the loss of the VAE process
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The folder containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/{}/'.format(tag))
    vae_lens_folder = os.path.join(folder, 'LENS/{}/VAE/'.format(tag))
    vae_source_folder = os.path.join(folder, 'SOURCE/{}/VAE/'.format(tag))
    os.makedirs(os.path.join(figure_folder, 'LOSS/'), exist_ok = True)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Lens
    with h5py.File(os.path.join(vae_lens_folder, 'DATA.hdf5'), 'r') as file:
        epoch_size = int(file['meta']['epoch_size'][...])
        
        all_loss_lens = file['history']['all_loss'][...]
        mean_loss_lens = file['history']['mean_loss'][...]
        scatter_loss_lens = file['history']['scatter_loss'][...]
        covariance_loss_lens = file['history']['covariance_loss'][...]
        divergence_loss_lens = file['history']['divergence_loss'][...]
        
        val_all_loss_lens = file['history']['val_all_loss'][...]
        val_mean_loss_lens = file['history']['val_mean_loss'][...]
        val_scatter_loss_lens = file['history']['val_scatter_loss'][...]
        val_covariance_loss_lens = file['history']['val_covariance_loss'][...]
        val_divergence_loss_lens = file['history']['val_divergence_loss'][...]
    epoch = numpy.arange(1, epoch_size + 1)
    
    figure, plot = pyplot.subplots(nrows = 1, ncols = 2, figsize = (16, 8))
    
    plot[0].plot(epoch, all_loss_lens, color = 'black', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{all}$')
    
    plot[0].plot(epoch, mean_loss_lens, color = 'purple', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{mean}$')
    
    plot[0].plot(epoch, scatter_loss_lens, color = 'blue', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{scatter}$')
    
    plot[0].plot(epoch, covariance_loss_lens, color = 'green', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{covariance}$')
    
    plot[0].plot(epoch, divergence_loss_lens, color = 'orange', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{divergence}$')
    
    plot[0].set_yscale('log')
    plot[0].set_xscale('linear')
    
    plot[0].set_ylim(1.0e-9, 1.0e+2)
    plot[0].set_xlim(1, epoch_size + 1)
    
    plot[0].set_title(r'$\mathrm{Train}$')
    plot[0].set_xlabel(r'$\mathrm{Epochs}$')
    plot[0].set_ylabel(r'$\mathrm{Losses}$')
    
    plot[1].plot(epoch, val_all_loss_lens, color = 'black', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{all}$')
    
    plot[1].plot(epoch, val_mean_loss_lens, color = 'purple', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{mean}$')
    
    plot[1].plot(epoch, val_scatter_loss_lens, color = 'blue', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{scatter}$')
    
    plot[1].plot(epoch, val_covariance_loss_lens, color = 'green', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{covariance}$')
    
    plot[1].plot(epoch, val_divergence_loss_lens, color = 'orange', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{divergence}$')
    
    plot[1].set_yscale('log')
    plot[1].set_xscale('linear')
    
    plot[1].set_ylim(1.0e-9, 1.0e+2)
    plot[1].set_xlim(1, epoch_size + 1)
    
    plot[1].set_yticklabels([])
    plot[1].legend(loc='upper right')
    
    plot[1].set_xlabel(r'$\mathrm{Epochs}$')
    plot[1].set_title(r'$\mathrm{Validation}$')
    
    figure.subplots_adjust(hspace=0.0, wspace = 0.00)
    figure.savefig(os.path.join(figure_folder, 'LOSS/FIGURE_LENS.pdf'), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Source
    with h5py.File(os.path.join(vae_source_folder, 'DATA.hdf5'), 'r') as file:
        epoch_size = int(file['meta']['epoch_size'][...])
        
        all_loss_source = file['history']['all_loss'][...]
        mean_loss_source = file['history']['mean_loss'][...]
        scatter_loss_source = file['history']['scatter_loss'][...]
        covariance_loss_source = file['history']['covariance_loss'][...]
        divergence_loss_source = file['history']['divergence_loss'][...]
        
        val_all_loss_source = file['history']['val_all_loss'][...]
        val_mean_loss_source = file['history']['val_mean_loss'][...]
        val_scatter_loss_source = file['history']['val_scatter_loss'][...]
        val_covariance_loss_source = file['history']['val_covariance_loss'][...]
        val_divergence_loss_source = file['history']['val_divergence_loss'][...]
    epoch = numpy.arange(1, epoch_size + 1)
    
    figure, plot = pyplot.subplots(nrows = 1, ncols = 2, figsize = (16, 8))
    
    plot[0].plot(epoch, all_loss_source, color = 'black', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{all}$')
    
    plot[0].plot(epoch, mean_loss_source, color = 'purple', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{mean}$')
    
    plot[0].plot(epoch, scatter_loss_source, color = 'blue', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{scatter}$')
    
    plot[0].plot(epoch, covariance_loss_source, color = 'green', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{covariance}$')
    
    plot[0].plot(epoch, divergence_loss_source, color = 'orange', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{divergence}$')
    
    plot[0].set_yscale('log')
    plot[0].set_xscale('linear')
    
    plot[0].set_ylim(1.0e-10, 1.0e+2)
    plot[0].set_xlim(1, epoch_size + 1)
    
    plot[0].set_title(r'$\mathrm{Train}$')
    plot[0].set_xlabel(r'$\mathrm{Epochs}$')
    plot[0].set_ylabel(r'$\mathrm{Losses}$')
    
    plot[1].plot(epoch, val_all_loss_source, color = 'black', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{all}$')
    
    plot[1].plot(epoch, val_mean_loss_source, color = 'purple', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{mean}$')
    
    plot[1].plot(epoch, val_scatter_loss_source, color = 'blue', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{scatter}$')
    
    plot[1].plot(epoch, val_covariance_loss_source, color = 'green', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{covariance}$')
    
    plot[1].plot(epoch, val_divergence_loss_source, color = 'orange', linestyle = '-', linewidth = 2.5, label = r'$\mathrm{divergence}$')
    
    plot[1].set_yscale('log')
    plot[1].set_xscale('linear')
    
    plot[1].set_ylim(1.0e-10, 1.0e+2)
    plot[1].set_xlim(1, epoch_size + 1)
    
    plot[1].set_yticklabels([])
    plot[1].legend(loc='upper right')
    
    plot[1].set_xlabel(r'$\mathrm{Epochs}$')
    plot[1].set_title(r'$\mathrm{Validation}$')
    
    figure.subplots_adjust(hspace=0.0, wspace = 0.00)
    figure.savefig(os.path.join(figure_folder, 'LOSS/FIGURE_SOURCE.pdf'), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Loss')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)