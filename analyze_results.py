from pathlib import Path
import os
import json
import numpy as np

def get_all_files(directory_path):
    p = Path(directory_path)
    # Recursively list all items and filter for files
    files = [item for item in p.rglob('*.json') if item.is_file()]
    return files

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

    num_success_per_rho = [0,0,0]
    rhos = ["p0.5", "p0.7", "p0.9"]
    for file in success_files:
        rho_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))
        rho = rho_dir.split("/")[-1]
        for i in range(len(rhos)):
            if rho == rhos[i]:
                num_success_per_rho[i] += 1
    
    print()
    print('Convergent simulations per rmin in {0.5, 0.7, 0.9}')
    print(f'rmin=0.5: {num_success_per_rho[0]}')
    print(f'rmin=0.7: {num_success_per_rho[1]}')
    print(f'rmin=0.9: {num_success_per_rho[2]}')

    num_fail_per_rho = [0,0,0]
    rhos = ["p0.5", "p0.7", "p0.9"]
    for file in fail_files:
        rho_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))
        rho = rho_dir.split("/")[-1]
        for i in range(len(rhos)):
            if rho == rhos[i]:
                num_fail_per_rho[i] += 1
    
    print()
    print('Non-convergent simulations per rmin in {0.5, 0.7, 0.9}')
    print(f'rmin=0.5: {num_fail_per_rho[0]}')
    print(f'rmin=0.7: {num_fail_per_rho[1]}')
    print(f'rmin=0.9: {num_fail_per_rho[2]}')

    num_success_per_ky = [0,0,0,0,0,0]
    kys = ["p1", "p2", "p3", "p4", "p5", "p6"]
    for file in success_files:
        ky = os.path.dirname(file).split("/")[-1].split("_")[0]
        for i in range(len(kys)):
            if ky == kys[i]:
                num_success_per_ky[i] += 1
    
    print()
    print('Convergent simulations per ky in {0.05, 0.3, 1.2, 2.5, 6, 10}')
    print(f'ky=0.05: {num_success_per_ky[1]}')
    print(f'ky=0.3 : {num_success_per_ky[0]}')
    print(f'ky=1.2 : {num_success_per_ky[2]}')
    print(f'ky=2.5 : {num_success_per_ky[3]}')
    print(f'ky=6 : {num_success_per_ky[4]}')
    print(f'ky=10 : {num_success_per_ky[5]}')

    num_fail_per_ky = [0,0,0,0,0,0]
    kys = ["p1", "p2", "p3", "p4", "p5", "p6"]
    for file in fail_files:
        ky = os.path.dirname(file).split("/")[-1].split("_")[0]
        for i in range(len(kys)):
            if ky == kys[i]:
                num_fail_per_ky[i] += 1
    
    print()
    print('Non-convergent simulations per ky in {0.05, 0.3, 1.2, 2.5, 6, 10}')
    print(f'ky=0.05: {num_fail_per_ky[1]}')
    print(f'ky=0.3 : {num_fail_per_ky[0]}')
    print(f'ky=1.2 : {num_fail_per_ky[2]}')
    print(f'ky=2.5 : {num_fail_per_ky[3]}')
    print(f'ky=6 : {num_fail_per_ky[4]}')
    print(f'ky=10 : {num_fail_per_ky[5]}')
    
    print()
    print('Convergence simulations per rmin AND per ky')
    for rho in rhos:
        print(f'rmin={rho}: ' + ('=' * 20))
        print()
        num_success_per_ky_per_rho = [0,0,0,0,0,0]
        for file in success_files:
            rho_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))
            rho_curr = rho_dir.split("/")[-1]
            if rho_curr != rho:
                continue
            ky = os.path.dirname(file).split("/")[-1].split("_")[0]
            for i in range(len(kys)):
                if ky == kys[i]:
                    num_success_per_ky_per_rho[i] += 1
        print(f'ky=0.05: {num_success_per_ky_per_rho[1]}')
        print(f'ky=0.3 : {num_success_per_ky_per_rho[0]}')
        print(f'ky=1.2 : {num_success_per_ky_per_rho[2]}')
        print(f'ky=2.5 : {num_success_per_ky_per_rho[3]}')
        print(f'ky=6 : {num_success_per_ky_per_rho[4]}')
        print(f'ky=10 : {num_success_per_ky_per_rho[5]}')
        print()

    print()
    print('Convergent simulations per physical case')
    for case in physical_cases:
        case_name = case.split('/')[1]
        print(f'Shot {case_name}: ' + ('=' * 20))
        print()
        num_success_per_case_per_rho = [0,0,0]
        for file in success_files:
            if case_name not in str(file):
                continue
            rho_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file))))
            rho = rho_dir.split("/")[-1]
            for i in range(len(rhos)):
                if rho == rhos[i]:
                    num_success_per_case_per_rho[i] += 1
        print(f'rmin=0.5: {num_success_per_case_per_rho[0]}')
        print(f'rmin=0.7: {num_success_per_case_per_rho[1]}')
        print(f'rmin=0.9: {num_success_per_case_per_rho[2]}')
        print()
