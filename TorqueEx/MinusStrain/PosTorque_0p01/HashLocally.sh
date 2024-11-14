#!/bin/bash
# bash to create folders, copy the input files, and sends jobs from these folder
echo 'mass start analysis with the parallelized version of the python hash script'

for i in {6,7}
do 
	cp HashMuOverTime_ProcessPoolEx_2Layers_forCluster.py run_$i/
	cd ./run_$i/
	rm *00000000*
	mkdir ProcessedData/
	python HashMuOverTime_ProcessPoolEx_2Layers_forCluster.py
	cd ../
done
