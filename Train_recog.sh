#!/bin/bash -l

#SBATCH
#SBATCH --job-name=SU_LOUO8
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=end
#SBATCH --mail-user=tkim60@jhu.edu

#### load and unload modules you may need
module restore mymodules
module load tensorflow/cuda-8.0/r1.0

echo "Using GPU Device:"
echo $CUDA_VISIBLE_DEVICES

python /home-4/tkim60@jhu.edu/scratch/dev/jigsaws/train_jigsaws.py --gpu=$CUDA_VISIBLE_DEVICES > /home-4/tkim60@jhu.edu/scratch/dev/jigsaws/LOUO8_SU_$SLURM_JOBID.log
echo "Finished with job $SLURM_JOBID"
