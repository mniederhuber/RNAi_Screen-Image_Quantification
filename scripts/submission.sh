#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH -t 2:00:00


if [ ! -d 'output/' ] 
then 
	mkdir output
 	mkdir output/maxProj
  mkdir output/mask
fi

if [ ! -d 'output/maxProj/' ]
then
 	mkdir output/maxProj
fi

if [ ! -d 'output/mask/' ]
then
  mkdir output/mask
fi

python scripts/imgParser.py
