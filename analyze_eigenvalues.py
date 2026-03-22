from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy

def fourier(eigenvalues, real_path, imag_path):
    print(eigenvalues.shape)
    eigenvalues = eigenvalues[0,0,:]
    reals = np.real(eigenvalues)
    imags = np.imag(eigenvalues)

    # Last half
    reals = reals[int(len(reals)/2):]
    imags = imags[int(len(imags)/2):]

    # Removes 1st order
    reals_detrended = scipy.signal.detrend(reals, type='linear')
    imags_detrended = scipy.signal.detrend(imags, type='linear')

    # Removes 0th order
    reals_detrended = scipy.signal.detrend(reals_detrended, type='constant')
    imags_detrended = scipy.signal.detrend(imags_detrended, type='constant')

    fft_real = np.fft.fft(reals_detrended)
    fft_imag = np.fft.fft(imags_detrended)

    # Need to de-trend on FFT (use scipy)
    # Especially on second half (/ two-thirds) of time series
    # At beginning, have oscillations where cgyro tries to converge early on
    # If eigenvalues oscillate, will find higher peak

    plt.plot(np.abs(fft_real))
    plt.xlabel('Frequency')
    plt.ylabel('Eigenvalue / Hz')
    plt.title('FFT on Reals')
    plt.savefig(real_path)
    plt.cla()
    plt.close()

    plt.plot(np.abs(fft_imag))
    plt.xlabel('Frequency')
    plt.ylabel('Eigenvalue / Hz')
    plt.title('FFT on Imags')
    plt.savefig(imag_path)
    plt.cla()
    plt.close()

def get_eigenvalues(h5_path):
    try:
        with h5py.File(h5_path, "r") as f:
            eigenvalues = f["eigenvalues"][:]
        return eigenvalues
    except FileNotFoundError:
        print(f'FileNotFoundError: {h5_path}')
        return None
    
def get_all_files(directory_path, suffix):
    p = Path(directory_path)
    # Recursively list all items and filter for files
    files = [item for item in p.rglob(f'*.{suffix}') if item.is_file()]
    return files

if __name__== "__main__":
    directory = './results_no_xi'
    print(f'Checking results in directory: {directory}')
    print("=" * 50)
    all_files = get_all_files(directory, 'json')

    success_files = []
    fail_files = []
    # Check successes and fails
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                meta = json.load(file)
            converged = meta['converged']
            if converged:
                success_files.append(file_path)
            else:
                fail_files.append(file_path)
        except FileNotFoundError:
            print(f"Error: The file {file} was not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    
    print(f'Successful runs: {len(success_files)}')
    print(f'Failed runs: {len(fail_files)}')

    success_index = int(np.floor(np.random.uniform() * len(success_files)))
    fail_index = int(np.floor(np.random.uniform() * len(fail_files)))

    success_h5 = os.path.join(os.path.dirname(success_files[success_index]), 'res.h5')
    fail_h5 = os.path.join(os.path.dirname(fail_files[fail_index]), 'res.h5')

    success_eig = get_eigenvalues(success_h5)
    fail_eig = get_eigenvalues(fail_h5)

    success_fft = fourier(success_eig, 'real_converged_eig_fft.png', 'imag_converged_eig_fft.png')
    fail_fft = fourier(fail_eig, 'real_nonconverged_eig_fft.png', 'imag_nonconverged_eig_fft.png')