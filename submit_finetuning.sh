#!/bin/bash
# set job name
#BSUB -J Wav2Vec2BirdsFinetuning
# set qeue to submit to
#BSUB -q gpuv100
# set number of nodes       
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
# set nodes distribution  
#BSUB -R "span[hosts=1]"
# set amount of memory ofr each core
#BSUB -R "rusage[mem=2GB]"  
# set wall time for the job execution
#BSUB -W 24:00       

# set email to receive notification
#BSUB -u andreismarco2002@gmail.com 
# notify at job start
#BSUB -B
# notify at job end
#BSUB -N

# Set output file
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

# Load Python module 
module load python3/3.10.12  

# Change to project directory
cd ~/Deep_learning_project

# Activate the virtual environment
source .venv_finetuning/bin/activate

# Run the training script
python3 -u scripts/train.py 
