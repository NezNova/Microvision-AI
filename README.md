# Automated Defect Inspection in Semiconductor SEM Images

## Repository Link

[https://github.com/NezNova/Microvision-AI]

## Description

This project applies deep learning to automatically classify SEM images of MEMS structures. A dataset of SEM images was organized and labeled into eight different MEMS-related classes, and a pretrained ResNet-50 model was fine-tuned to learn and distinguish their characteristic visual features.
This classification task represents a first step toward teaching AI to understand and interpret MEMS SEM images. The long-term goal is to extend the approach beyond structure recognition toward automatic defect detection and defect classification, supporting faster and more efficient SEM image analysis and quality assessment.

### Task Type

Image Classification 

### Results Summary

#### Best Model Performance
- **Best Model:** Pretrained Microsoft ResNet-50 fine-tuned on cropped SEM images
- **Evaluation Metric:** Validation accuracy, weighted F1-score, precision, and recall
- **Final Performance:** 92.97% validation accuracy and 0.927 weighted F1-score

#### Model Comparison
- **Baseline Performance:** ResNet-50 image-only baseline achieved 92.97% validation accuracy
- **Improvement Over Baseline:** No later experiment improved over the baseline; the unfreeze-from-start experiment matched the baseline at 92.97%
- **Best Alternative Model:** Image + metadata model achieved 92.43% validation accuracy and 0.920 weighted F1-score
- 
#### Key Insights
- **Most Important Features:** Cropped SEM image pixels from the gray microscopy region
- **Model Strengths:** Strong classification performance on majority classes such as Bond-Pad-Array, 3d_edge, Electrode, close_up_line, and waveguide
- **Model Limitations:** Minority classes such as label and microfluidic had very few real examples, and OCR-extracted metadata was noisy and did not improve final accuracy
- **Business Impact:** This project is a first step toward automating MEMS SEM image analysis. The model can reduce manual sorting effort by classifying SEM images into structural categories. In the future, this pipeline could be extended so AI can recognize MEMS structures, detect defects, and classify different defect types in SEM images.
  
## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/exploratory_data_analysis.ipynb)**
3. **[Baseline Model](2_BaselineModel/baseline_model.ipynb)**
4. **[Model Definition and Evaluation](3_Model/model_definition_evaluation)**
5. **[Presentation](4_Presentation/README.md)**

## Cover Image

![Project Cover Image](CoverImage/cover_image.png)
