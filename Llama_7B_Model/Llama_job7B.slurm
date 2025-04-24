#!/bin/sh
#SBATCH --time=03:00:00  # Time limit (hours:minutes:seconds)
#SBATCH --job-name=hgfctrial_job  # Job name
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --nodes=1  # Number of nodes
#SBATCH --partition=normal  # Partition to use
#SBATCH --output=hgfctrial-%j.out  # Standard output file
#SBATCH --error=hgfctrial-%j.err  # Standard error file

# Change to the directory where the script is located
cd /home/tuq33671/VirusProjectStella2025/Llama_7B_Model

# Load any required modules (if applicable)
# Example: module load python/3.8

# Run the Python script
python Llama7BModel.py