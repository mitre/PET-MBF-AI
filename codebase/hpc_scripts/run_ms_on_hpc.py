#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys

BASE_DIR = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'

description = """
    Launches a gridsearch job on the HPC for SVM or RF by submitting the job
    to the SLURM queue with sbatch. This function will create directories for any 
    directory path passed as an argument that does not exist at the time the 
    function is run. The function will create files for: 

        the result of the grid search; 
        the dictionary of hyperparameter values to be used in the grid search (as a JSON file); 
        the log file for the SLURM job; 
        and the environment variables for the SLURM job. 

    This function launches a gridsearch job to SLURM with sbatch. It specifies
    how much memory, how many cpus, and what time limit are requested. It
    instructs SBATCH to run the program ms_on_hpc.py, and it passes command
    line arguments to ms_on_hpc.py specifying the model, dataset, and
    various paths for input and output files. For this script to run, 
    both the script its self and ms_on_hpc.py must be located in 
    'e-emagin-pet/codebase/hpc_scripts', and data files
    PET_2008_to_2019_deidentified_exclusions_applied_train.xlsx and 
    PET_2008_to_2019_deidentified_exclusions_applied_test.xlsx must be saved in
    TODO: UPDATE THIS WITH NEW FILES AND LOCATIONS.

"""


# Parse command line arguments
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--model', 
                    type=str, 
                    choices=['svm', 'rf', 'mlp', 'unet'],
                    help="""The model to run gridsearch for. Valid options are
                            'svm', 'rf', 'mlp', 'unet'""")
parser.add_argument('--problem',
                        type=str,
                        choices=['norm_abn', 'localization'],
                        help="""The classification problem to solve. Valid
                              choices include 'norm_abn' or 'localization'""")
parser.add_argument('--data_dir',
                    required=True,
                    type=str,
                    help="""Path to the directory contianing the dataset""")
parser.add_argument('--dataset',
                    type=str,
                    choices=['3_vessel', '17_segment', 'polar_plot'],
                    help="""The dataset to train on. Valid choices include 
                            '3_vessel', '17_segment', and 'polar_plot'""")
parser.add_argument('--hp_spec', 
                    type=str,
                    help="""The path to the JSONified dictionary of hyperparameters
                          outlining hyperparameters and values to gridsearch over.
                          This is in the format of the param_grid argument
                          of scikit learn's model_selection.GridSearchCV""")
parser.add_argument('--job_name', 
                    type=str,
                    help='job name')
parser.add_argument('--results_dir',
                    nargs='?', 
                    type=str,
                    help="""Path to directory to save results.""")
parser.add_argument('--log_dir',
                    nargs='?', 
                    type=str, 
                    help="""Path to the directory where log file will be saved.""")
parser.add_argument('--time', 
                    nargs='?', 
                    type=str,
                    default="05:00", 
                    help="Time limit for job. Default value is 05:00")
parser.add_argument('--mem',
                    nargs='?',
                    type=str, 
                    default="10GB", 
                    help='Memory for job. Default value is 10GB')
parser.add_argument('--cpus_per_task', 
                    nargs='?', 
                    type=int, 
                    default=1, 
                    help='CPUs to request. Default value is 1')
parser.add_argument('--gpus_per_node', 
                    nargs='?', 
                    type=int, 
                    default=0, 
                    help='GPUs per node to request. Default value is 0')
parser.add_argument('--num_samples',
                    required='ho' in sys.argv,
                    type=int,
                    default=None, 
                    help="""When nn_search_alg is 'ho, num samples is the 
                            number of trials to run in Bayesian optimization.
                            When nn_search_alg is 'gs' this is the number
                            of times to resample the gridsearch (i.e.
                            num_samples=2 will run the gridsearch twice).
                            This can be helpful as random initializations
                            neural networks can alter results. Required 
                            when nn_search_alg is 'ho'""") 
parser.add_argument('--epochs',
                    required=('mlp' in sys.argv) or ('unet' in sys.argv),
                    type=int, 
                    default=None, 
                    help="""Number of epochs to run in random search. Required
                            if model is 'mlp' or 'unet', otherwise not used.""")
parser.add_argument('--nn_search_alg',
                    required='mlp' in sys.argv,
                    type=str, 
                    default=None,
                    choices=['ho', 'gs'],
                    help="""The algorithm to use for hyperparmeter tuning for
                          a neural network. Valid options include 'ho' for
                          HyperOpt and 'gs' for grid search.""")
parser.add_argument('--local',
                    action='store_true',
                    help="""Use to run the hyperparameter search locally rather
                            than launching a new job with SLURM""")


args = parser.parse_args()
model = args.model
data_dir = args.data_dir
dataset = args.dataset
hp_spec_path = args.hp_spec
job_name = args.job_name
results_dir = args.results_dir
log_dir = args.log_dir
time = args.time
mem = args.mem
cpus = args.cpus_per_task
gpus = args.gpus_per_node
num_samples = args.num_samples
epochs = args.epochs
nn_search_alg = args.nn_search_alg
local = args.local
problem = args.problem

# build string with the command line arguments to be passed to ms_on_hpc.py
ms_script_args = f"--model {model} --problem {problem} --data_dir {data_dir} --dataset {dataset} --hp_spec {hp_spec_path} --results_dir {results_dir} --job_name {job_name} "

optional_params = [num_samples, epochs, gpus, nn_search_alg]
optional_param_flags = ['--num_samples', '--epochs', '--gpus_per_node', 
                        '--nn_search_alg']
for param, flag in zip(optional_params, optional_param_flags):
    if param:
        ms_script_args = ms_script_args + " " + flag + " " + str(param)

ms_script_path = BASE_DIR + 'codebase/hpc_scripts/ms_on_hpc.py'
get_port_path = BASE_DIR + 'codebase/hpc_scripts/get_port.py'
output_file = BASE_DIR +  'codebase/hpc_scripts/start_ray_output.txt'
singularity_path = BASE_DIR + 'Singularity/ray_nvidia.sif'

slurm_script_sklearn = \
f"""EOF
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --output={log_dir}/%u-%x-job%j.out
#SBATCH --mem={mem}
#SBATCH --gpus-per-node={gpus}
#SBATCH --export=ALL
#SBATCH -N1 --cpus-per-task {cpus}

singularity exec --bind /q:/q {singularity_path} python3 {ms_script_path} {ms_script_args}

# Print Environment Variables to File
env > {log_dir}/slurm_env_{job_name}.txt

echo "Job completed."

EOF
"""

slurm_script_ray = \
f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --output={log_dir}/%u-%x-job%j.out
#SBATCH --mem={mem}
#SBATCH --gpus-per-node={gpus}
#SBATCH --export=ALL
#SBATCH -N1 --cpus-per-task {cpus}

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
node1=${{nodes_array[0]}}

# run 'hostname --ip-address' on $node1 to get the ip address of node1
head_node_ip=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address)

unused_port=$(python {get_port_path})
ip_head="${{head_node_ip}}:${{unused_port}}"
redis_password=$(uuidgen)
export ip_head

# start ray on node1 on port 6379
# ray start --> The command will print out the address of the Redis server that was started
srun --nodes=1 --ntasks=1 -w $node1 singularity exec --nv --bind /q:/q {singularity_path} ray start --block --head  --node-ip-address="$head_node_ip" --port=$unused_port --include-dashboard False --num-cpus "${{SLURM_CPUS_PER_TASK}}" --num-gpus "${{SLURM_GPUS_PER_NODE}}" --redis-password=$redis_password & # Starting the head
# srun --nodes=1 --ntasks=1 -w $node1 singularity exec --nv --bind /q:/q ~/ray_nvidia.sif ray start --block --head --port=6379 --redis-password=$redis_password & # Starting the head
sleep 5

singularity exec --nv --bind /q:/q {singularity_path} python3 {ms_script_path} {ms_script_args} --redis_password $redis_password

# Print Environment Variables to File
env > {log_dir}/slurm_env_{job_name}.txt

echo "Job completed."
"""


# for local job, just run ms_script
if local:
    cmd = f"python {ms_script_path} {ms_script_args} --local"
# for sklearn models, run the script directly 
elif model == 'svm' or model == 'rf':
    slurm_script = slurm_script_sklearn
    cmd = "sbatch <<" + slurm_script 
# for keras models (using ray for hyperparameter tuning) write slurm script
# to a temporary file, and then launch temporary script with sbatch
elif (model == 'mlp') or (model == 'unet'):
    slurm_script = slurm_script_ray
    run_path = BASE_DIR + 'codebase/hpc_scripts/'
    tmp_run_script_name = 'run_ms_tmp.sh'
    tmp_run_script = run_path + tmp_run_script_name
    with open(tmp_run_script, "w") as outfile:
        outfile.write(slurm_script_ray)

    cmd = f"sbatch {tmp_run_script}"

proc = subprocess.Popen(cmd,
    shell=True,
    stdout = subprocess.PIPE,
    stderr = subprocess.PIPE,
)
stdout, stderr = proc.communicate()
print(stdout.decode("utf-8"))
print(stderr.decode("utf-8"))


