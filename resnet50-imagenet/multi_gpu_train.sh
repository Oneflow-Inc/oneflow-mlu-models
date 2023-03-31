python3 -m oneflow.distributed.launch --nproc_per_node 2 main.py -a resnet50 --multiprocessing-distributed -j8 -b 128
