# Chest X-ray Dataset (Raw Data)

This project is based on a publicly available chest X-ray (CXR) dataset for COVID-19 and pneumonia analysis.

The original dataset was developed by a multidisciplinary research team from Qatar University (Doha, Qatar) and the University of Dhaka (Bangladesh), in collaboration with medical doctors and international partners. The dataset aggregates chest X-ray images from multiple publicly accessible sources and previously published studies.

## Dataset composition

The original dataset includes the following classes:
- COVID-19
- Normal
- Viral Pneumonia
- Lung Opacity (Non-COVID lung infection)

In this project, **only the following classes were used**:
- COVID-19  
- Normal  
- Viral Pneumonia  

The *Lung Opacity* category was not included in the current modeling pipeline and is therefore excluded from training and evaluation.

## Image characteristics and preprocessing considerations

- Image format: PNG  
- Original image resolution (as distributed in this version of the dataset): **256 × 256 pixels**

During preprocessing, all images were resized to **224 × 224 pixels** in order to align with the input requirements of **MobileNetV2**, which was pretrained on ImageNet using this standard resolution.  
No additional image enhancement techniques (e.g. CLAHE) were applied at this stage to avoid introducing artificial contrast patterns in chest X-ray images.

## Data availability

Due to dataset size and licensing considerations, **raw images are not included in this repository**.

Users interested in reproducing this work should download the dataset directly from the original sources listed below and organize the images following the folder structure described in the main project documentation.

## Original data sources

COVID-19 chest X-ray images were collected from several publicly available datasets, including:
- BIMCV COVID-19 dataset  
- German medical school repositories  
- SIRM, GitHub repositories, Kaggle, and other public collections  

Normal and Viral Pneumonia images were obtained from:
- RSNA Pneumonia Detection Challenge  
- Kaggle Chest X-ray Pneumonia dataset  

The full list of original sources and dataset aggregation details are provided in the official dataset documentation.

## License

The dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license, as specified by the original authors.

This license allows reuse and adaptation of the data provided that appropriate credit is given to the original creators.

## Citation

When using this dataset, the original authors request citation of the following works:

- Chowdhury, M.E.H. et al., *Can AI help in screening Viral and COVID-19 pneumonia?* IEEE Access, 2020.  
- Rahman, T. et al., *Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images*, arXiv:2012.02238.
