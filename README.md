## About

A PSPNet([Pyramid Scene Parsing Network](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)) implementation with Tensorflow.

## Set up

+ Prepare for dataset

    + download Cityscape from [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)
    + The directory should be as follows:
        ```
        + Cityscape
           + img_test.txt	
           + img_train.txt	
           + img_val.txt
           + anno_test.txt
           + anno_train.txt	
           + anno_val.txt	
           + leftImg8bit
                + train
                + val
                + test
           + gtFine	
                + train
                + val
                + test
        ```
    
+ Download the pretrained model
    + download pretrained resnet101 weight from [http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz)
    + download the trained weight from [here]() **if you want to inference and evaluate the model**.
    

> python train.py

## Inference 

> python predict.py

## Evaluation

> python evaluate.py

## Results

## Re-Train to produce the result

```
python cityscape.py

pytho train.py

```