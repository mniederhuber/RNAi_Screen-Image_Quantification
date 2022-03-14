#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=2g
#SBATCH -t 2:00:00


if [ ! -d 'output/' ] 
then 
	mkdir output
fi

python scripts/imgParser.py
