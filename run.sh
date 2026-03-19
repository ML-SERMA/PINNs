#!/bin/bash

# Exemple de script de soumission qui exécute la commande "hostname".
# de la partition amdq avec la qos normal

#SBATCH -n 1                                  # Nombre de taches
#SBATCH -c 24 ## 24 32 ##48                   # Nombre de coeurs
#SBATCH -p gpuq_h100                          # Partitions adaptées au job  
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=7jour                           # QOS
#SBATCH -J Pinns                           # Nom du job
#SBATCH --time=7-00:00:00                     # Nom du job
#SBATCH -o gpuPinns%j.o                       # Fichier de sortie
#SBATCH -e gpuPinns%j.e                       # Fichier d'erreur
#SBATCH --mail-user=minh-hieu.do@cea.fr       # Adresse email
#SBATCH --mail-type=begin,end,fail            # Envoi d'email à l'exécution, fin et échec du job


export OMP_NUM_THREADS=4      # Controls number of CPU threads for NumPy/OpenMP
export MKL_NUM_THREADS=4      # Same for Intel MKL (if used)

echo "Checking GPU availability..."
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

python3 -m benchmarks.multi_group.exp_TWIGL_2D --no-scaling-loss
exit 0
