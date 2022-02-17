# Pneumonia X-Ray Image Classification
---

# Overview
---
<img src='images/lung.png' width=300 align='right'/>
<br>
Pneumonia is an infection that inflames the air sacs in our lungs. Many germs can cause pneumonia -- the most common are bacteria and viruses in the air we breathe. Your body usually prevents these germs from infecting your lungs, but sometimes these germs can overpower your immune system, even if your health is generally good. Pneumonia is the world’s leading cause of death among children under 5 years of age.
An x-ray exam will allow your doctor to see your lungs, heart and blood vessels to help determine if you have pneumonia. When interpreting the x-ray, the radiologist will look for white spots in the lungs (called infiltrates) that identify an infection.

Using Convolutional Neural Networks we can train a model that identifies whether a patient has pneumonia given a chest x-ray. Our final model has a 95% accuracy on our testing images.

# Business Understanding
--- 
Machine Learning has shown impressive accuracy in medical imaging. Creating a predictive model for identifying pneumonia from x-ray images provides medical professionals with a system that can automate the process for accurate identification of pneumonia, saving time and resources.

As with many predictive models used in medicine, we would rather minimize false negatives (patient has pnumonia but the model classifies their x-ray as normal) than false positives (patient doesn't have pnumonia but the model classifies x-ray as pnumonia). This is because we are dealing with the lives and health of humans. Therefore we want to create a model that minimizes false negatives.

# Data Understanding
---
This dataset contains 5,856 validated Chest X-Ray images. The images categorized into two subsets -- Pneumonia and Normal. The images are split into training, validation, and testing sets.

This dataset comes from Kermany et al and can be found at https://data.mendeley.com/datasets/rscbjbr9sj/3

All x-rays were initially screened for quality control and graded by three expert physicians before being cleared for training the AI system.

# Modeling
---

# Final Evaluation
--- 

# Conlusion
--- 
Our final model has an accuracy of 95% on predicting whether an x-ray image contains evidence of Pneumonia (viral or bacterial). As we can see from our confusion matrix, our model shows to classify more false positives ('classifies as pnumonia, but not') than false negatives ('classifies as normal, but actually pneumonia'). This alligns strongly with our business problem so we will consider this model to be very efficient in classifying pneumonia from chest x-rays.

# Next Steps
--- 
This type of methodology can be extremely useful in the identification of infections and adnormalities in medical imaging (not just x-ray but MRI, CT's, etc..). The use of maching learning techniques has the potential to be extremely useful in the medical field, but it also has the potential to be harmful. We have to be mindful with the types of scenarios we are using maching learning and AI for and be socially responsible when diploying models. 


# Repository Structure
---
```
├── data                # contains original datasets and saved models
│   ├── chest_xray      # X-Ray images split into train, test, validation
│   ├── models          # Saved tensorflow model
├── images 
├── README.md
├── pneumonia-image-classification.ipynb
└── pneumonia_image_classification.pdf

```