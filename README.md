# NOTE:
***This repository has been upgraded, a new deep learning model has been provided(works only on full body and top wear clothes) has been released. Algorithm migration has been carried for the ease segmentation in version1.1***

# Version 1.1
![results](https://github.com/anish9/Fashion-AI-segmentation/blob/master/result/collage.png)

# Requirements
#### Tensorflow 2.0-alpha
#### OpenCV
#### Python3.6

# Inference
****download the pretrained model and directly run****
```
python run.py mydress.jpg

```

> [download_pretrained](https://drive.google.com/open?id=14vTYmsHjUYv3VPo1Byrecs3NQuvJo89t)

****Snippet to integrate anywhere****
```
api    = fashion_tools(f,saved)
image_ = api.get_dress()
cv2.imwrite("out.png",image_)

```


# Version 1.0 
![results](https://user-images.githubusercontent.com/25944164/35455349-8ada7410-02f7-11e8-905e-84dad8ee01df.jpg)
*A New Approach by using the Blend of Image-Processing Technique and Deep-Learning Algorithm to Segment any Fashion and e-commerce Retail Images*.
The code can be used for any industry on any images and the core algortithm is 'grab-cut algorithm" with the blend of Deep-Learning Convolutional Neural Networks. The Repo is designed in a preview way and its limited for fashion Images with auto-segmenting Top-wear clothes(Example: Tshirt, shirts) and Full-body clothes(salwar,gowns, shirt-pants-shoes)*

## Image-Processing Resource 
https://en.wikipedia.org/wiki/GrabCut
## Deep Learning
https://en.wikipedia.org/wiki/Deep_learning
# Package Requirements
**1.Python<enter>**
  
**2.OpenCV 3.1.0<enter>**
  
**3.Keras with tensorflow backend<enter>**
  
**4.Pandas<enter>**
  
**5.NumPy<enter>**

# Demo v1.0
#### Note: Demo Annotation shouldn't be replaced, adding new will not enable the code adaptation to new classes of images.(*The demo phase classes : Fashion full-body, Top-wear*)
```
1.*clone* the Repo to your local pc ensuring that all the package requirements satisfied.<enter>
  
2.Run the code from the terminal **python fashion.py image1.jpg /Users/demo/save** <enter>
  
3.argument1 -- *image_name -- image1.jpg*, argument2 -- *save_directory -- /Users/demo/*
```
4.Visualize the results in your save_directory.



