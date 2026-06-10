# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline Model Type:** pretrained Microsoft ResNet-50 fine-tuned on cropped SEM images
- **Rationale:** A pretrained ResNet-50 model was chosen as the baseline because it is a well-established convolutional neural network for image classification tasks. Using a pretrained model allows transfer learning from large-scale image datasets, which is useful because the SEM dataset is relatively small and imbalanced. ResNet-50 provides a strong and reliable starting point for evaluating whether cropped SEM image features are sufficient for classifying the 8 MEMS categories. Its performance can later be compared with improved models using metadata, additional data, or different architectures.

### Model Performance
- **Evaluation Metric:** Accuracy, precision, recall, and F1-score were used. Accuracy was used as the main overall metric, while precision, recall, and F1-score were used to evaluate class-level performance.
- **Performance Score:** The baseline ResNet-50 model achieved 92.97% validation accuracy. The weighted average F1-score was approximately 0.927.
- **Cross-Validation Score:** Cross-validation was not performed. The model was evaluated using a fixed train/validation split with 2,867 training images and 555 validation images.

### Evaluation Methodology
- **Data Split:** The dataset was split into training and validation sets using approximately an 80/20 split. The prepared dataset contained 2,867 training images and 555 validation images. The validation set was not augmented, so it provided a more realistic evaluation of model performance. In addition, the trained model was applied to 1,000 unlabeled SEM images as an external inference set to inspect prediction behavior on new data. These 1,000 images were not used to calculate final accuracy because ground-truth labels were not available.
- **Evaluation Metrics:** The model was evaluated using accuracy, precision, recall, and F1-score. Accuracy was used to measure overall classification performance across all SEM image classes. Precision and recall were included because the dataset is imbalanced, meaning accuracy alone may hide poor performance on minority classes. F1-score was used as a balanced metric that combines precision and recall, making it useful for comparing class-level performance.

### Metric Practical Relevance
Accuracy measures the overall percentage of SEM images that were classified correctly. In this project, the baseline ResNet-50 model achieved 92.97% validation accuracy, meaning the model correctly classified most validation images. Practically, high accuracy indicates that the model can reduce manual sorting effort and help researchers quickly organize SEM images into MEMS structure categories.

Precision measures how often the model is correct when it predicts a specific class. This is important because a low-precision class would mean many images are being placed into the wrong category. For example, if the model predicts Electrode, high precision means those images are likely truly electrodes, which helps avoid incorrect dataset organization.

Recall measures how many images of a true class the model successfully finds. This is important when missing certain structures would be costly. For example, low recall for microfluidic means the model may fail to identify many microfluidic images, so researchers would still need manual review for that class.

F1-score combines precision and recall into one balanced metric. This is especially useful because the dataset is imbalanced: some classes have hundreds of examples, while others have very few. A strong F1-score means the model is not only accurate overall but also performs reasonably well across different classes.

In this project, the weighted average F1-score was approximately 0.927, which shows strong overall performance. However, minority classes such as microfluidic and label had weaker scores due to limited examples, so these classes should be reviewed carefully before using the model for fully automated classification.

## Next Steps
This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.
