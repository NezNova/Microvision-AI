# Dataset Characteristics

**Notebook:** input/1_DatasetCharacteristics/exploratory_data_analysis.ipynb

## Dataset Information

### Dataset Source
- **Dataset Link:** http://hdl.handle.net/11304/7b330bf0-50e5-400a-8bc1-3f6a072f5ffa 
- **Dataset Owner/Contact:**

  Dataset: NFFA-EUROPE SEM Dataset

  Creators: Rossella Aversa, Mohammad Hadi Modarres, Stefano Cozzini, Regina Ciancio

  Publisher: NFFA-EUROPE Project / EUDAT B2SHARE

  License: CC BY 4.0

  Contact: rossella.aversa@nffa.eu

### Dataset Characteristics
- **Number of Observations:** Approximately 2800 SEM images (2D grayscale microscopy images)
- **Number of Features:** The trained ResNet model uses cropped SEM images resized to 224 x 224 RGB pixels, corresponding to 150,528 pixel-level input features per image. The dataset has 8 target classes.

Metadata was extracted separately but was not used in the current baseline model.

### Target Variable/Label
- **Label Name:**  8 classes: 3d_edge, Bond-Pad-Array, cantilever, close-up-line, Electrode, label, microfluidic, and waveguide.
- **Label Type:** Classification
- **Label Description:** Each label corresponds to a specific MEMS structural feature identified from SEM images. The task is multi-class classification, where the model predicts the MEMS component category shown in the image.
- **Label Values:**

  3d_edge: SEM images showing three-dimensional edge-like microstructures or elevated patterned boundaries.

  Bond-Pad-Array: Images containing arrays of bond pads used for electrical connection and packaging of MEMS devices.

  cantilever: Images showing cantilever-based MEMS structures such as suspended beams or flexible arms.

  close-up-line: High-magnification images focused on narrow line structures, interconnects, or patterned linear features.

  Electrode: Images containing electrode structures used for electrical actuation, sensing, or signal transmission.

  label: Images primarily containing textual markings, fabrication information, identifiers, or metadata visible in the SEM image.

  microfluidic: Images showing microfluidic channels, chambers, or fluid transport structures in MEMS devices.

  waveguide: Images containing waveguide structures used for optical or signal-guiding applications.

- **Label Distribution:** The original dataset is imbalanced across 8 SEM image classes. The largest classes are Bond-Pad-Array with 825 images, close_up_line with 660 images, and Electrode with 628 images. Smaller classes include waveguide with 129 images, cantilever with 36 images, microfluidic with 18 images, and label with only 2 images. To reduce imbalance during training, minority classes were augmented in the training set.

### Feature Description
[Provide a brief description of each feature or group of features in your dataset. If you have many features, group them logically and describe each group. Include information about data types, ranges, and what each feature represents.]

**Example format:**
- **Feature Group 1 (cropped SEM image pixels):** The main input features are pixel values from the cropped SEM image region. The bottom microscope information ribbon was removed before training. Each image was resized to 224 x 224 pixels and converted to RGB format with 3 channels, giving 224 x 224 x 3 = 150,528 pixel-level input features per image. Pixel values are numeric and normalized before being passed into the pretrained ResNet-50 model.
- **Feature Group 2 (image label / target class):** Each image has one categorical target label based on its folder name. The dataset contains 8 classes: 3d_edge, Bond-Pad-Array, cantilever, close_up_line, Electrode, label, microfluidic, and waveguide. This label is the output the model learns to predict.
- **Feature Group 3 (metadata features, extracted but not used in baseline model):** Microscope metadata was extracted from the white ribbon at the bottom of the original SEM images using OCR. These features include scale value, accelerating voltage EHT_kV, working distance WD_mm, signal type, stage tilt angle, stage Z position, magnification, and aperture size. These are numeric or categorical metadata fields and were saved separately for possible future model improvement, but the current baseline ResNet model was trained only on the cropped SEM image pixels.

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](input/1_DatasetCharacteristics/exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment
