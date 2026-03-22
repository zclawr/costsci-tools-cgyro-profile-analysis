from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Time cost dimensions: (rho, ky, 3)
# Last dim is [mean, std, max]

# Seaprate np.arrays for both converged and non-converged runs

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

    rhos = ["p0.5", "p0.7", "p0.9"]
    kys = ["p1", "p2", "p3", "p4", "p5", "p6"]

    nky = len(kys)
    nrho = len(rhos)

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

    success_cost_stats = np.zeros((nrho, nky, 3))
    success_cost_stats[:, :, 0] = np.mean(success_costs, axis=-1)
    success_cost_stats[:, :, 1] = np.std(success_costs, axis=-1)
    success_cost_stats[:, :, 2] = np.max(success_costs, axis=-1)

    SEC_TO_HOURS = 3600
    print('Time cost statistics for converged runs: ' + '=' * 50)

    print('DETAILED STATISTICS: ' + '*' * 50)
    for i in range(len(rhos)):
        print(f'rmin = {rhos[i]}: ' + '-' * 20)
        print()
        print(f'ky=0.05: mean={success_cost_stats[i, 1, 0]}s, std={success_cost_stats[i,1,1]}s, max={success_cost_stats[i,1,2] / SEC_TO_HOURS} hrs')
        print(f'ky=0.3: mean={success_cost_stats[i, 0, 0]}s, std={success_cost_stats[i,0,1]}s, max={success_cost_stats[i,0,2] / SEC_TO_HOURS} hrs')
        print(f'ky=1.2: mean={success_cost_stats[i, 2, 0]}s, std={success_cost_stats[i,2,1]}s, max={success_cost_stats[i,2,2] / SEC_TO_HOURS} hrs')
        print(f'ky=2.5: mean={success_cost_stats[i, 3, 0]}s, std={success_cost_stats[i,3,1]}s, max={success_cost_stats[i,3,2] / SEC_TO_HOURS} hrs')
        print(f'ky=6: mean={success_cost_stats[i, 4, 0]}s, std={success_cost_stats[i,4,1]}s, max={success_cost_stats[i,4,2] / SEC_TO_HOURS} hrs')
        print(f'ky=10: mean={success_cost_stats[i, 5, 0]}s, std={success_cost_stats[i,5,1]}s, max={success_cost_stats[i,5,2] / SEC_TO_HOURS} hrs')
        print()

    print('HIGH LEVEL STATISTICS: ' + '*' * 50)
    for i in range(len(rhos)):
        print(f'rmin = {rhos[i]}: ' + '-' * 20)
        print(f'Mean: {np.mean(success_cost_stats[i,:,0] / SEC_TO_HOURS)} hrs')
        print(f'Std: {np.mean(success_cost_stats[i,:,1] / SEC_TO_HOURS)} hrs')
        print(f'Max: {np.max(success_cost_stats[i,:,2] / SEC_TO_HOURS)} hrs')
        print()
    
    worst_case_cost = np.max(np.max(np.max(success_costs, axis=-1), axis=-1), axis=-1)
    
    NUM_LLMS = 5
    NUM_RUNS_PER_CHECKOUT_PER_RHO = 5

    rho_1_cost = NUM_RUNS_PER_CHECKOUT_PER_RHO * np.max(success_cost_stats[0,:,2])
    rho_2_cost = NUM_RUNS_PER_CHECKOUT_PER_RHO * np.max(success_cost_stats[1,:,2])
    rho_3_cost = NUM_RUNS_PER_CHECKOUT_PER_RHO * np.max(success_cost_stats[2,:,2])

    worst_case_llm_cost = NUM_LLMS * (rho_1_cost + rho_2_cost + rho_3_cost)

    print(f'Estimating worst case LLM checkout time: ' + '*' * 50)
    print(f'Number of LLMs: {NUM_LLMS}')
    print(f'Number of runs per LLM checkout per rmin: {NUM_RUNS_PER_CHECKOUT_PER_RHO}')
    print()
    print(f'WORST CASE LLM BENCHMARK COST: {worst_case_llm_cost / SEC_TO_HOURS / 24} days')
    print(f'TOTAL CHECKOUTS PER LLM: {NUM_RUNS_PER_CHECKOUT_PER_RHO * nrho}')

    # Cluster by rmin, ky
    ky_values = [0.3, 0.05, 1.2, 2.5, 6, 10]
    rho_values = [0.5, 0.7, 0.9]

    # Omega determines ion-scale mode (< 1), electron scale mode (> 1)
    # Sign of frequency determines ion mode or electron mode (this is consistent across cases bc of IPCCW/BTCCW signs)
    # At each rmin, check which ky converged where, choose the cases closest to centroids

    # Cluster around omega for (rho, ky, omega)
    # Found in left column of out.tglf.freq

    # Check 4th line of out.tglf.info (MUST be INFO: (CGYRO) Ion direction: omega > 0)
    # If this is not the case, throw out for clustering