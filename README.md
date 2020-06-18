# Distiller-3D

[![N|Solid](https://photos.angel.co/startups/i/7668151-5d88d918640632929007cc03461a8e4b-medium_jpg.jpg)](https://ultrainstinct.ai/)

##### Key features
  - Prune any model of any architecture 3D or 2D CNN 
  - Magic

# Various Approches followed:
- ###  [Taylor ranking](https://arxiv.org/pdf/1611.06440.pdf)
  - This paper taught how we should rank filters or kernels in the CNNs. They gave a greedy formula of ranking filters.
  - There is a github [repo](https://github.com/jacobgil/pytorch-pruning) for this paper
  - I extended this repo for 3D CNNs and prunned C3D like architectures and fine tuned on ucf101 and ucf11
  - **Results:** 
    > **Model size:** 
    > 135 MB prunned
    > 350 MB unprunned
    > **Run time on cuda:**
    > 5.7 sec for unprunned
    > 0.81 sec for prunned
    > cuda batch 32:
    > 22.78 sec for prunned
    > 28.9 sec for unprunned
    > **Run time on CPU:**
    > 1 batch 10 epcochs 3.36 sec
    > 2 batch 10 epochs 6.18 sec
    > 3 batch 10 epochs 11.22 sec
    > 11 batch 10 epochs 34.55 sec

   - **Major Files**
Branch -> master
       <center> <img src="Major Files in master.png"></center>

- ###  [Resnet Like architectures](https://arxiv.org/pdf/1608.08710.pdf)
    - This paper taught how much we can filter prune architectures like VGG-16, Resnet-110. 
    - They showed "How to solve the data dependency problem in Resudual connection".
    - They conduct sensitivity analysis(how much we can prune individual layer) for CNNs including ResNets.
    - This screenshot shows the data dependecy problem of resiudal block.
<center> <img src="Residual prunning.png"></center>

    - This screenshot shows how much each layer is valued in entire network. Also shows the relationship between different layers in resiudal block.
<center> <img src="Sensitivity analysis .png"></center>

    - **Code**
    [This](https://github.com/eeric/channel_prune) is the only open sourced implementation I found to understand this process. **Note-** There were few bugs in this code, I made a pull request to run this code seamlessly. If it is not merged yet Find [here](https://github.com/kunalgoyal9/channel_prune) :)
    
- ### [Distiller](https://gitlab.com/ultrainstinctAI/liteinstinct/-/tree/distiller-3d)
    - Through this framework we can prune any model. Also there is good [tutorial](https://nervanasystems.github.io/distiller/usage.html) from nervana.
    - There is an app in **distiller-3d/examples/classifier_compression/e.py** called classifier_compression which uses **yaml** config files in **distiller/examples/pruning_filters_for_efficient_convnets/slowfast_ucf101_filter_rank.yaml**
    - ##### Major Changes made:
        - I have deleted major assert statements in the thinning.py, policy.py, pruning/ranked_structures_pruner.py files to perform 3D convolution.
        - I have added model in **distiller-3d/distiller/models/__init__.py** file and dataset for UCF101 in **distiller-3d/distiller/apputils/data_loaders.py** files. All of these follows the similar pattern to that with slowfast repo.
    - ##### Command:
        - Before running the command, we need to add Slowfast repo in PYTHONPATH as 
        ``` export PYTHONPATH=/workspace/Kugos/SlowFast/slowfast:$PYTHONPATH ```
        - ```python3 compress_classifier.py -a=slowfast_ucf101 -p=50 --epochs=70 --lr=0.1 --DATASET_DIR="." --compress=../pruning_filters_for_efficient_convnets/slowfast_ucf101_filter_rank.yaml --resume-from=logs/slowfast_92_percent/ --batch-size=2 --reset-optimizer --vs=0 ```
        - **NOTE:** I have inplicitly specified the ``` --DATASET_DIR ``` in **dataset .py** file.
        - 
