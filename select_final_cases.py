from sklearn.cluster import KMeans
from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tarfile

# Import the mplot3d toolkit
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D 

# Time cost dimensions: (rho, ky, 3)
# Last dim is [mean, std, max]

# Seaprate np.arrays for both converged and non-converged runs

def get_all_files(directory_path):
    p = Path(directory_path)
    # Recursively list all items and filter for files
    files = [item for item in p.rglob('*.json') if item.is_file()]
    return files

def get_omega_from_tar(archive_name):
    # Open the tar archive
    try:
        # Use mode='r:*' for transparent compression handling (recommended)
        with tarfile.open(archive_name, mode='r:*') as tar:
            # Get a TarInfo object for the specific file
            member = tar.getmember('out.cgyro.info')

            # Access the file content as a file-like object without extracting to disk
            with tar.extractfile(member) as file_obj:
                if file_obj is not None:
                    # Read the content (initially bytes) and decode if it's text
                    content_bytes = file_obj.read()
                    content_string = content_bytes.decode('utf-8') # Assuming UTF-8 encoding
                    REQUIRED_SUBSTRING = 'INFO: (CGYRO) Ion direction: omega > 0'
                    flip_sign = False
                    if REQUIRED_SUBSTRING not in content_string:
                        flip_sign = True

                    member2 = tar.getmember('out.cgyro.freq')
                    with tar.extractfile(member2) as file_obj2:
                        if file_obj2 is not None:
                            # Read the content (initially bytes) and decode if it's text
                            content_bytes = file_obj2.read()
                            content_string = content_bytes.decode('utf-8') # Assuming UTF-8 encoding
                            omega = float(content_string.split('\n')[-2].strip().split(' ')[0].strip())
                            if flip_sign:
                                omega = -omega
                            return omega
                        else:
                            print("out.cgyro.freq could not be read (e.g., might be a directory or link).")
                else:
                    print("out.cgyro.info could not be read (e.g., might be a directory or link).")

    except tarfile.ReadError:
        print(f"Error reading the tar file {archive_name}. Check file format and compression.")
    except KeyError:
        print(f"File out.cgyro.info not found in the archive.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f'Occurred on file with contents:')
        print(content_string)

    return None

if __name__== "__main__":
    directory = './results'
    print(f'Checking results in directory: {directory}')
    print("=" * 50)
    all_files = get_all_files(directory)
    num_files = len(all_files)
    print()
    print(f'Found {num_files} total simulations')

    # Check for physical cases (shot + rho)
    physical_cases = set()
    for file in all_files:
        case_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))
        physical_cases.add(case_dir)
    
    physical_cases_ky = set()
    for file in all_files:
        case_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))
        ky = os.path.dirname(file).split("/")[-1].split("_")[0].strip()
        case_dir_with_ky = case_dir + '/' + ky
        physical_cases_ky.add(case_dir_with_ky)

    num_physical_cases = len(physical_cases_ky)
    print(f'Found {num_physical_cases} total physical cases ((shot @ timeslice) * rmin * ky)')

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

    print()
    num_success = len(success_files)
    num_fail    = len(fail_files)
    print(f'Total convergent simulations: {num_success}')
    print(f'Total non-convergent simulations: {num_fail}')

    rhos = ["p0.5", "p0.7", "p0.9"]
    kys = ["p1", "p2", "p3", "p4", "p5", "p6"]

    ky_values = [0.3, 0.05, 1.2, 2.5, 6, 10]
    rho_values = [0.5, 0.7, 0.9]
    
    nky = len(kys)
    nrho = len(rhos)

    # Check for pre-existing processed case file
    try:
        success_cases_pre = np.load('success_cases.npy')
        print('Loaded processed case file')
    except FileNotFoundError:
        print('No processed case file found, creating now')
        success_cases_pre = np.array([True])

    if success_cases_pre.all() == True:
        success_costs = np.zeros((nrho, nky, num_success))
        for i in range(len(success_files)):
            file_path = success_files[i]
            rho_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
            rho = rho_dir.split("/")[-1]
            ky = os.path.dirname(file_path).split("/")[-1].split("_")[0]
            for j in range(len(rhos)):
                if rho == rhos[j]:
                    for k in range(len(kys)):
                        if ky == kys[k]:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                meta = json.load(file)
                                success_costs[j, k, i] = meta['cost']

        # Cluster by rmin, ky
        success_cases = []
        i = 0
        for file in success_files:
            i = i + 1
            if i % 100 == 0:
                print(f'{i/len(success_files) * 100}%: Processing file {i}/{len(success_files)}')
            file_index = success_files.index(file)
            try:
                with open(file, 'r', encoding='utf-8') as file_obj:
                    meta = json.load(file_obj)
                cost = meta['cost']
                rho_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))
                rho = rho_dir.split("/")[-1].strip()
                ky = os.path.dirname(file).split("/")[-1].split("_")[0].strip()

                rho_i = rhos.index(rho)
                ky_i = kys.index(ky)

                rho_val = rho_values[rho_i]
                ky_val = ky_values[ky_i]
                omega = get_omega_from_tar(os.path.join(os.path.dirname(file), 'cgyro_outputs.tar.gz'))
                if omega == None:
                    continue
                success_cases.append([rho_val, ky_val, omega, file_index, cost])
            except FileNotFoundError:
                print(f"Error: The file {file} was not found.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        
        success_cases = np.array(success_cases)
        np.save('success_cases.npy', success_cases)
    else:
        success_cases = success_cases_pre
        print(f'Using loaded case file')
    
    costs = success_cases[:, 4]
    file_idxs = success_cases[:, 3]
    omegas = success_cases[:, 2]
    kys = success_cases[:, 1]
    rmins = success_cases[:, 0]

    ion_cases = success_cases[np.where(omegas > 0), :].squeeze()
    elec_cases = success_cases[np.where(omegas < 0), :].squeeze()

    shot_costs = dict()
    print()
    for i in range(len(success_files)):
        file_path = success_files[i]
        shot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))).split('/')[-1].strip()
        with open(file_path, 'r', encoding='utf-8') as file:
            meta = json.load(file)
            cost = meta['cost']
        if shot in shot_costs.keys():
            shot_costs[shot] = np.append(shot_costs[shot], [cost])
        else:
            shot_costs[shot] = np.array([cost])
    
    worst_case_shot_costs = dict()
    print(f'Worst case shot costs:')
    for shot in shot_costs.keys():
        worst_case_shot_costs[shot] = np.max(shot_costs[shot])
        print(f'{shot}: {worst_case_shot_costs[shot]/3600}hrs')

    # Select final cases based on Alessandro's suggestion as follows:
    # If twelve cases is our limit I would suggest ion and electron modes at ky = 0.05 0.3 1.2 2.5 and rmin = 0.9, 
    # which is 8 in total, to capture the spectrum. Plus ion and electron modes at ky = 0.3 at rmin = 0.3 and rmin = 0.7, 
    # which is an additional 4, to capture the radial dependence of ky=0.3, which is one of the modes that generates most transport.

    samples_per_rmin = [2, 2, 8]
    kys_per_rmin = [[0.3], [0.3], [0.05, 0.3, 1.2, 2.5]]
    final_cases = []

    for i in range(len(rho_values)):
        ion_rmin = ion_cases[np.where(ion_cases[:,0] == rho_values[i])]
        elec_rmin = elec_cases[np.where(elec_cases[:,0] == rho_values[i])]
        print(f'rmin={rho_values[i]}' + '=' * 20)
        for j in range(len(kys_per_rmin[i])):
            ky = kys_per_rmin[i][j]
            print(f'ky={ky}')
            ion_rmin_ky = ion_rmin[np.where(ion_rmin[:,1] == ky)]
            elec_rmin_ky = elec_rmin[np.where(elec_rmin[:,1] == ky)]

            best_ion_shot = None
            best_elec_shot = None
            ion_min_worst_case_shot_cost = -1
            elec_min_worst_case_shot_cost = -1

            for ion_file_idx in ion_rmin_ky[:,3]:
                file = success_files[int(ion_file_idx)]
                shot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))).split('/')[-1].strip()
                worst_case = worst_case_shot_costs[shot]
                if ion_min_worst_case_shot_cost == -1 or worst_case < ion_min_worst_case_shot_cost:
                    ion_min_worst_case_shot_cost = worst_case
                    best_ion_shot = shot

            unique_elec_shots = set()
            for elec_file_idx in elec_rmin_ky[:,3]:
                file = success_files[int(elec_file_idx)]
                shot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))).split('/')[-1].strip()
                worst_case = worst_case_shot_costs[shot]
                if elec_min_worst_case_shot_cost == -1 or worst_case < elec_min_worst_case_shot_cost:
                    elec_min_worst_case_shot_cost = worst_case
                    best_elec_shot = shot
            
            print(f'Best electron shot: {best_elec_shot} (worst case cost = {elec_min_worst_case_shot_cost / 3600}hrs)')
            print(f'Best ion shot: {best_ion_shot} (worst case cost = {ion_min_worst_case_shot_cost / 3600}hrs)')
            print()
            final_cases.append([rho_values[i], ky, best_elec_shot, best_ion_shot])

    # Final cases acquired!  

    # Shape: (rmin, ky, best_elec_shot, best_ion_shot)
    final_cases = np.array(final_cases)
    print(final_cases)
    print()
    print('SEPARATE PLASMA TO RUN FOR FINAL CHECKOUT ' + '*' * 50)
    final_plasmas = set() # Each entry is "rmin + shot"
    for i in range(final_cases.shape[0]):
        final_plasmas.add(f"{final_cases[i,0]} + {final_cases[i,2]}")
        final_plasmas.add(f"{final_cases[i,0]} + {final_cases[i,3]}")
    
    final_plasmas = np.array(list(final_plasmas))
    print(final_plasmas)