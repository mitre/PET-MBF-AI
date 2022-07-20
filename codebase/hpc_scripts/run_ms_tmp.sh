#!/bin/bash
#SBATCH --job-name=unet_pp_r6_Larger_bs
#SBATCH --time=100:00:00
#SBATCH --output=/q/PET-MBF/output_df/polar_plot/localization/unet/logs/%u-%x-job%j.out
#SBATCH --mem=40GB
#SBATCH --gpus-per-node=1
#SBATCH --export=ALL
#SBATCH -N1 --cpus-per-task 3

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
node1=${nodes_array[0]}

# run 'hostname --ip-address' on $node1 to get the ip address of node1
head_node_ip=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address)

unused_port=$(python /q/PET-MBF/dberman/e-emagin-pet/codebase/hpc_scripts/get_port.py)
ip_head="${head_node_ip}:${unused_port}"
redis_password=$(uuidgen)
export ip_head

# start ray on node1 on port 6379
# ray start --> The command will print out the address of the Redis server that was started
srun --nodes=1 --ntasks=1 -w $node1 singularity exec --nv --bind /q:/q /q/PET-MBF/dberman/e-emagin-pet/Singularity/ray_nvidia.sif ray start --block --head  --node-ip-address="$head_node_ip" --port=$unused_port --include-dashboard False --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --redis-password=$redis_password & # Starting the head
# srun --nodes=1 --ntasks=1 -w $node1 singularity exec --nv --bind /q:/q ~/ray_nvidia.sif ray start --block --head --port=6379 --redis-password=$redis_password & # Starting the head
sleep 5

singularity exec --nv --bind /q:/q /q/PET-MBF/dberman/e-emagin-pet/Singularity/ray_nvidia.sif python3 /q/PET-MBF/dberman/e-emagin-pet/codebase/hpc_scripts/ms_on_hpc.py --model unet --problem localization --data_dir /q/PET-MBF/data --dataset polar_plot --hp_spec /q/PET-MBF/output_df/polar_plot/localization/unet/hp_specs/unet_pp_r6_Larger_bs_hp_spec.joblib --results_dir /q/PET-MBF/output_df/polar_plot/localization/unet/results --job_name unet_pp_r6_Larger_bs  --num_samples 500 --epochs 700 --gpus_per_node 1 --nn_search_alg ho --redis_password $redis_password

# Print Environment Variables to File
env > /q/PET-MBF/output_df/polar_plot/localization/unet/logs/slurm_env_unet_pp_r6_Larger_bs.txt

echo "Job completed."
