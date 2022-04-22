# Gradient-based refined class activation map for weakly supervised object localization


Weakly supervised object localization locates objects based on the localization map generated from the classification network. However, most existing methods utilize the information of the target class to locate objects based on the feature map of a single image, which ignores both the relationships of interclass and intra-class. In this work, we propose a Gradient-based Refined Class Activation Map (GRCAM) approach to achieve more accurate localization. Two kinds of gradients are applied to reveal the relationships of inter-class and intra-class during the testing stage. First, we exploit the gradients of the classification loss function concerning the feature map to enhance class-specific information. The gradients of classification loss reveal the connection among the predicted probabilities of all classes. Second, we design a regression function that refers to the loss between the pseudo-bounding box coordinates containing category consistency and the predicted coordinates generated from the localization map. The predicted coordinates are revised by the gradients of the regression function. The gradients of the regression function reveal the consistency within a class. Despite the apparent simplicity, we demonstrate the advantages of GRCAM on ILSVRC and CUB-200-2011 in extensive experiments. Especially, on ILSVRC dataset, the proposed GRCAM achieves a new state-of-the-art Top-1 localization error of 42.94%.

# The proposed GRCAM method

![image](https://user-images.githubusercontent.com/103172926/164704790-0417d6ab-8f07-41b2-b243-9f02dc5a30f5.png)

# Trained model

* Download [the trained models](https://drive.google.com/drive/folders/1dLa44PWKYsYVvM9hfWqIVvikx2CCF2IO?usp=sharing) on ImageNet at GoogleDrive.

# Pre-requisites
  
 * Python = 3.8.10
 * Tensorflow = 2.5.0
 * Tensorpack = 0.11
 * Pytorch = 1.8.0
 * Torchvision = 0.9.0
 * opencv-python == 4.5.2.52 
 * Pillow == 8.2.0
 * tqdm == 4.60.0

# Localization
  
![image](https://user-images.githubusercontent.com/103172926/164706678-8f3be781-9fca-4951-8dd6-2ec31cd5ab2b.png)
![image](https://user-images.githubusercontent.com/103172926/164706755-4c887dcf-9df0-4619-b86d-6d331eefe203.png)

# Test
  
```
python3 Generate_predicted_bbox.py
cd pseudo_bbox
python3 rebuild_val.py
python3 main.py
cd ..
python3 revise_bbox.py
```
  
# Citation
  
If you find the code helpful, please consider to cite the paper:
  
```
@article{hui2022gradient,
  title={Gradient-based Refined Class Activation Map for Weakly Supervised Object Localization},
  author={Hui, Wenjun and Tan, Chuangchuang and Gu, Guanghua and Zhao, Yao},
  journal={Pattern Recognition},
  pages={108664},
  year={2022},
  publisher={Elsevier}
}
```
