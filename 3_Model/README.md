# Model Definition and Evaluation

**[Notebook](model_definition_evaluation)**
## Model Development Summary

The baseline image-only ResNet-50 model remained the best-performing model. Three additional experiments were conducted to test whether metadata or hyperparameter changes improved performance.

| Experiment | Change Tested | Validation Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| Baseline ResNet-50 | Cropped SEM image only, LR=3e-5, 20 epochs | 92.97% | 0.927 |
| Image + Metadata | Added OCR microscope metadata | 92.43% | 0.920 |
| Lower LR + More Epochs | LR=1e-5, 30 epochs | 90.81% | 0.907 |
| Unfreeze From Start | freeze-backbone-epochs changed from 3 to 0 | 92.97% | 0.927 |


## Final Model Choice

The final selected model is the pretrained Microsoft ResNet-50 fine-tuned on cropped SEM images only. Adding metadata did not improve performance, likely because OCR-extracted metadata was noisy and incomplete. Lowering the learning rate reduced performance, and changing the freeze schedule did not provide a clear advantage.

## Key Conclusion

Further improvement is more likely to come from better dataset quality, more real examples for minority classes, and clearer class definitions rather than simple model-complexity or hyperparameter changes.
