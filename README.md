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
        | Filename | Info |
        | ------ | ------ |
        | finetune_3d-UCF11.py | This file uses UCF11 dataset with C3D architecture. We can specify the percentage we want our model to be prunned. I tried pruning percentage ranging from 10 to 90 percent and got same results. As the dataset has only 11 classes, so 90 percent prunned model also worked fine. |
        | finetune_3d-UCF101.py | Similar to finetune_3d-UCF11.py, this file prune C3D like architectures for UCF101 datasets. |
        | prune .py | This file is a dependency of finetune_3d* file. This defines the algorithm of pruning(majorly how we needs to change the output and input channels of the prunned layer and the following layer).  |

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

