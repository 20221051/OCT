# Predicting Visual Acuity Using OCT Images
Predicting a patient's visual acuity solely based on retinal images is a challenging task.<br/> 
Therefore, we conducted a study to train a deep learning model that predicts visual acuity using only OCT (Optical Coherence Tomography) images of patients.<br/>

This study aims to: <br/>

Train a deep learning model to predict visual acuity from OCT images. <br/>
Utilize retinal data from patients with macular degeneration to aid in disease diagnosis and prediction. <br/>
By leveraging simple imaging techniques, we hope to not only predict visual acuity but also contribute to the diagnosis and prediction of diseases such as macular degeneration. <br/>

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
  
# Results
<img src='./images/gradcam.png'>  <br/>

We used VGG16 model. <br/>

This can be seen as an encouraging result, <br/>
as it is not easy for actual doctors to accurately adjust vision by looking only at retinal images. <br/>

In particular, the lower part of the OCT is a very important part for vision, <br/>
and we confirmed that the model learned well through gradcam.

The code can be available in main.py.

