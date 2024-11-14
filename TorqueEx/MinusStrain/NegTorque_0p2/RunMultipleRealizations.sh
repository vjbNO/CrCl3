#!/bin/bash
# bash to create folders, copy the input files, and sends jobs from these folder
echo 'creating directories, copying files and running simulations many times'

for i in {1..10}
do 
	mkdir run_$i
	cp *geo run_$i/
	cp *mat run_$i/
	cp *ucf run_$i/
	cp *chk run_$i/
	cp input run_$i/
	cp vampire-parallel run_$i/
	cp vmprQueue.sh run_$i/
	cd ./run_$i/
	echo 'sim:integrator-random-seed='$RANDOM >> input
	sbatch ./vmprQueue.sh
	cd ../
done