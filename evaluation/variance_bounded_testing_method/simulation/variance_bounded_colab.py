from multiprocessing.pool import ThreadPool
import subprocess
import sys
import os.path

# globals
launched = 0
done = 0
n_runs = 0

def print_status():
    print('Launched: {}/{} Done: {}'.format(launched, n_runs, done), end='\r', flush=True)

def launch_scenario(index):
    global launched
    global done
    launched += 1
    print_status()
    p = subprocess.run(
        ['python', '/content/sim_loop/evaluation/variance_bounded_testing_method/simulation/variance_bounded_colab_execution.py'] + ['--osc'] + [f'/content/sim_loop/scenarios/variance_bounded/{str(index)}_cut-in.xosc'] + ['--fixed_timestep'] + ['0.05'] + ['--headless'] + ['--window'] + ['60 60 800 400'] + ['--logfile_path'] + [f'../{str(index)}_log.txt'] + [str(param_values[idx,2])],
        stdout=subprocess.DEVNULL
    )

    done += 1
    print_status()


if __name__ == '__main__':

    n_runs = len(param_values)
    print_status()

    with ThreadPool() as p:
        p. map(launch_scenario, range(n_runs))

    print()