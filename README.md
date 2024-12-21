# Pruning Bird Sound Classification Models

[Our Project Synopsis](https://docs.google.com/document/d/1LIKyPUPawW4ij3PQ5Zt5m0szr0RyrzrwKHArcgJQetw/edit?tab=t.0)

[Drive folder containing the data](https://drive.google.com/drive/folders/1cy33llQGKs591txlE3DTkuCwVmv9jYJP?usp=drive_link)

In terms of execution the project was divided into:
- data processing, which is computationally intensive and needed to be done in batches to avoid problems memory allocation using pyarrow (01_data_processing.py).
- convexity analysis of the base model, which lead to us to prune the base model (using 07_manual_pruning.py 12 different version from just the feature extractor to the full model), perform the finetuning (01_finetuning.py) and the validation (03_model_validation.py).
- finetuning of the base wav2vec (01_finetuning.py) and similarity analysis, the results of which guided the process of post-training pruning (pruning was done both with the 05_pruning.py scripts, which also performs validation, and then with the 07_manual_pruning.py script, which produced models that were tested with the 03_model_validation.py script).
- training and testing of a CNN.

All the computation was carried out using the HPC Tesla A100 queue (gpua100). 

As can be seen the project makes intense use of many scripts that where run multiple times for testing and evalutation of many of models, with the main results being the comparison of the performance of these models which are summarized in the validation_results folder. For this reason we could not find a suitable way of creating a notebook capable of summarizing the results of the project.

