# OCT

# Data
10 OCT images for each of 240 patients.  
## Example and Distribution
<img src='./images/oct.png' width="400" height="200"/> <img src='./images/distribution.png' width="400" height="200"/>  

# AUC of 5-Class Multi Classification
<img src='./images/auc.png'>  
5 class  <br/>
1 : <0.1  <br/>
2 : <=0.3  <br/>
3 : <=0.5  <br/>
4 : <0.8  <br/>
5 : >=0.8 <br/>
  
## Results
<img src='./images/gradcam.png'>  <br/>

We used VGG16 model. <br/>

This can be seen as an encouraging result, <br/>
as it is not easy for actual doctors to accurately adjust vision by looking only at retinal images. <br/>

In particular, the lower part of the retina is a very important part for vision, <br/>
and we confirmed that the model learned well through gradcam.

