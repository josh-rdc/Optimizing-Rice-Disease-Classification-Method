# Optimizing rice disease classification: A comparative study of different color, shape and texture features

## Overview
![GraphicalAbstract/GraphicalAbstract.png](GraphicalAbstract/GraphicalAbstract.png)

Rice, often considered the staple of life in the Philippines, plays an integral role in the country's culture and sustenance. With a staggering domestic consumption of 16.5 million metric tons of milled rice in 2023, the reliance on this crop is undeniable. However, global rice production faces significant challenges from diseases which can cause yield losses of up to 70%, particularly when they strike early in the crop's life cycle. Effective rice disease management, therefore, relies heavily on early classification and forecasting systems that enable strategic interventions to minimize yield losses, reduce environmental impacts, and support sustainable agriculture. The integration of image processing and machine-driven systems is emerging as a vital solution to address the challenges of accurate disease classification.

This project introduces a novel and efficient approach for classifying 13 rice diseases, as compared to the usual 2-3 diseases focused by local and international studies, by analyzing entire images without requiring complex pre-processing and segmentation steps. By leveraging global image features, including color, shape, and texture, the study trained SVM and ANN models, achieving classification accuracies of **86.23%** and **83.93%**, respectively. 

The detailed dataset, methodology, and results will be discussed in the following sections.
- [Dataset](#dataset)
- [Feature Extraction and Processing](#feature-extraction-and-processing)
- [Training and Tuning](#training-and-tuning)
- [Experiments](#experiments)
- [Results and Discussion](#results-and-discussion)
- [Conclusion and Recommendations](#conclusion-and-recommendations)
- [References](#references)
- [Citation](#citation)

## Dataset
The dataset used in this study was retrieved from [Kaggle, originally from Omdena’s Local Chapter project- Creating a Rice Disease Classifier using Open Source Data and Computer Vision](https://www.kaggle.com/datasets/shrupyag001/philippines-rice-diseases), comprises two folders, namely `extra_resized_raw_images` and `resized_raw_images`, each containing diverse images of rice plants (paddy images, zoomed-in images, processed images) with samples shown below. 
| **paddy**               | **zoomed-in** | **processed** |
|---------------------------|------------------|-----------------------|
| ![assets/paddy.jpg](Assets/paddy.jpg) | ![assets/zoomed.jpg](Assets/zoomed.jpg) | ![assets/preprocessed.jpg](Assets/preprocessed.jpg) |

All images are standardized to a dimensionality of 224 × 224 pixels. Within the dataset, there are 13 distinct rice diseases categorized into three groups:
1. **Fungal**: Affects the leaf blade, sheath, stem, node, and panicle.
2. **Bacterial**: Affects the leaf blade.
3. **Viral**: Affects the leaf blade and sheath.

The `resized_raw_images` folder, which consists of original images for all 14 classes, including 13 diseases and 1 healthy class, was utilized. This subset has undergone cleaning and evaluation by other users, ensuring the removal of near-duplicate images from different classes. The dataset, as summarized in below, is balanced across the classes, providing a comprehensive representation of the different rice plant conditions.

| **Disease**               | **Class Number** | **Number of Images** |
|---------------------------|------------------|-----------------------|
| Bacterial leaf blight     | 0                | 97                    |
| Bacterial leaf streak     | 1                | 99                    |
| Bakanae                   | 2                | 100                   |
| Brown spot                | 3                | 100                   |
| Grassy stunt virus        | 4                | 100                   |
| Healthy rice plant        | 5                | 100                   |
| Narrow brown spot         | 6                | 98                    |
| Ragged stunt virus        | 7                | 100                   |
| Rice blast                | 8                | 98                    |
| Rice false smut           | 9                | 99                    |
| Sheath blight             | 10               | 98                    |
| Sheath rot                | 11               | 91                    |
| Stem rot                  | 12               | 100                   |
| Tungro virus              | 13               | 100                   |

## Feature Extraction and Processing
The jpeg images were translated to BGR images before converting to the needed color space to extract the input features for the model. The conversion was done by making use of OpenCV’s [CV2](https://docs.opencv.org/3.4/d8/d01/group\_\_imgproc\\\_\_color\_\_conversions.html). Specific details of the color spaces used to extract each feature are detailed in the table below:

 | **Feature Type**               | **Color Space** | **Details** |
|---------------------------|------------------|-----------------------|
| Texture |	Grayscale |	Derived from the Grey Level Co-occurrence Matrix (GLCM) which were extracted using the [Mahotas library](https://mahotas.readthedocs.io/en/latest/features.html). These features encodes patterns such as contrast, correlation, energy, and homogeneity. |
| Color Histogram |	RGB, HSV, LAB | Provides pixel distribution across color spaces which were split into individual channels. Calculated by computing the number of pixels for each histogram bins. |
| Color Moments |	RGB, HSV, LAB |	Offers compact color representation through statistical measures such as mean, variance, skewness, and kurtosis. Each value were computed for each channel using [NumPy](https://numpy.org/). |
| Zernike and Legendre Moments | Grayscale, HSV	| Orthogonal moments known to their effectiveness as   invariant shape descriptors. Vectors were extracted using [Mahotas Features](https://mahotas.readthedocs.io/en/latest/features.html) for Zernike and [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/\\scipy.special.legendre.html) for Legendre. |

## Training and Tuning

## Experiments

## Results and Discussion

## Conclusion and Recommendations

### References

- *M.A. Gumapac.* Rice: A Filipino Constant, *Bar Digest,* 25(10), 2011, pp. 1337-1342.
- *Index Mundi.* Philippines Milled Rice Domestic Consumption by Year, Retrieved: April 5, 2014.
- *S. Pandey.* International Rice Research Institute Rice in the Global Economy: Strategic Research and Policy Issues for Food Security, ISBN: 9789712202582, LCCN: 2011308303, 2010.
- *Nalley, L., Tsiboe, F., Durand-Morat, A., Shew, A., Thoma, G.* Economic and Environmental Impact of Rice Blast Pathogen (*Magnaporthe oryzae*) Alleviation in the United States. *PLoS ONE,* 11(12), e0167295, 2016.
- *C.M. Vera Cruz, I. Ona, N.P. Castilla, and R. Opulencia.* Bacterial Blight of Rice. Retrieved from [IRRI Rice Doctor](http://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/bacterial-blight).
- *Taohidul Islam et al.* A Faster Technique on Rice Disease Detection Using Image Processing of Affected Area in Agro-Field. *Patuakhali Science and Technology University,* 2018.
- *Kaundal R, Kapoor AS, Raghava GPS.* Machine learning techniques in disease forecasting: a case study on rice blast prediction. *BMC Bioinformatics,* 7:485, 2006.
- *Pagani V et al.* A high-resolution, integrated system for rice yield forecasting at district level. *Agricultural Systems,* 2018.
- *M. Sheerin Banu \& K. Nallaperumal.* Analysis of color feature extraction techniques for pathology image retrieval systems. *International Journal of Computational Technology and Applications,* 2:1930–1938, 2010.
- *D.S. Zhang, Md. M. Islam, and G.J. Lu.* A review on automatic image annotation techniques. *Pattern Recognition,* vol. 45, no. 1, 2012, pp. 346-362.
- *Novak, Carol L., and Steven A. Shafer.* Anatomy of a color histogram. *CVPR,* vol. 92, 1992.
- *Keen, Noah.* Color moments. *School of Informatics, University of Edinburgh,* 2005, pp. 3-6.
- *Haralick, R. M., Shanmugam, K., \& Dinstein, I.* Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics,* 3(6), 1–12, 1973.
- *L. Han, M. S. Haleem, and M. Taylor.* A novel computer vision-based approach to automatic detection and severity assessment of crop diseases. *Science and Information Conference (SAI),* pp. 638–644, 2015.
- *M.K. Hu.* Visual pattern recognition by moment invariants. *IRE Transactions on Information Theory,* 8(2):179–187, 1962.
- *S. X. Liao.* Image Analysis by Moments. *The Department of Electrical and Computer Engineering, University of Manitoba,* Winnipeg, Manitoba, Canada, 1993.
- *M.R. Teague.* Image analysis via the general theory of moments. *J. Opt. Soc. Am.,* 70(8):920–930, 1980.
- *Shah, J. P., Prajapati, H. B., \& Dabhi, V. K.* A Survey on Detection and Classification of Rice Plant Diseases. *Proceedings of the IEEE,* 2016.
- *Orillo, John William \& Valenzuela, Ira \& Cruz, Jennifer.* Identification of Diseases in Rice Plant (*Oryza Sativa*) using Back Propagation Artificial Neural Network. *10.1109/HNICEM.2014.7016248,* 2014.
- *Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., \& Talwalkar, A.* Journal of Machine Learning Research, 18(185), 1–52, 2018.
- *Hosaini, S. J., Alirezaee, S., Ahmadi, M., \& Makki, S. V.-A. D.* Comparison of the Legendre, Zernike and Pseudo-Zernike Moments for Feature Extraction in Iris Recognition. *2013 5th International Conference on Computational Intelligence and Communication Networks (CICN),* pp. 483–487. DOI: 10.1109/CICN.2013.54, 2013.
- *Hupkens, Th. M.* Legendre Moments of Colour Images. *Netherlands Defence Academy,* 2009.


-- in-progress

<!-- 
![Resources/[AI%20201]%20MINIPROJECT%20(1).png](Resources/[AI%20201]%20MINIPROJECT%20(1).png)
![Resources/[AI%20201]%20MINIPROJECT%20(2).png](Resources/[AI%20201]%20MINIPROJECT%20(2).png)
![Resources/[AI%20201]%20MINIPROJECT%20(3).png](Resources/[AI%20201]%20MINIPROJECT%20(3).png)
![Resources/[AI%20201]%20MINIPROJECT%20(4).png](Resources/[AI%20201]%20MINIPROJECT%20(4).png)
![Resources/[AI%20201]%20MINIPROJECT%20(5).png](Resources/[AI%20201]%20MINIPROJECT%20(5).png)
![Resources/[AI%20201]%20MINIPROJECT%20(6).png](Resources/[AI%20201]%20MINIPROJECT%20(6).png)
![Resources/[AI%20201]%20MINIPROJECT%20(7).png](Resources/[AI%20201]%20MINIPROJECT%20(7).png)
![Resources/[AI%20201]%20MINIPROJECT%20(8).png](Resources/[AI%20201]%20MINIPROJECT%20(8).png)
![Resources/[AI%20201]%20MINIPROJECT%20(9).png](Resources/[AI%20201]%20MINIPROJECT%20(9).png)
![Resources/[AI%20201]%20MINIPROJECT%20(10).png](Resources/[AI%20201]%20MINIPROJECT%20(10).png)
![Resources/[AI%20201]%20MINIPROJECT%20(11).png](Resources/[AI%20201]%20MINIPROJECT%20(11).png)
![Resources/[AI%20201]%20MINIPROJECT%20(12).png](Resources/[AI%20201]%20MINIPROJECT%20(12).png)
![Resources/[AI%20201]%20MINIPROJECT%20(13).png](Resources/[AI%20201]%20MINIPROJECT%20(13).png)
![Resources/[AI%20201]%20MINIPROJECT%20(14).png](Resources/[AI%20201]%20MINIPROJECT%20(14).png)
![Resources/[AI%20201]%20MINIPROJECT%20(15).png](Resources/[AI%20201]%20MINIPROJECT%20(15).png)
![Resources/[AI%20201]%20MINIPROJECT%20(16).png](Resources/[AI%20201]%20MINIPROJECT%20(16).png)
![Resources/[AI%20201]%20MINIPROJECT%20(17).png](Resources/[AI%20201]%20MINIPROJECT%20(17).png)
![Resources/[AI%20201]%20MINIPROJECT%20(18).png](Resources/[AI%20201]%20MINIPROJECT%20(18).png)
![Resources/[AI%20201]%20MINIPROJECT%20(19).png](Resources/[AI%20201]%20MINIPROJECT%20(19).png)
![Resources/[AI%20201]%20MINIPROJECT%20(20).png](Resources/[AI%20201]%20MINIPROJECT%20(20).png)
![Resources/[AI%20201]%20MINIPROJECT%20(21).png](Resources/[AI%20201]%20MINIPROJECT%20(21).png)
![Resources/[AI%20201]%20MINIPROJECT%20(22).png](Resources/[AI%20201]%20MINIPROJECT%20(22).png)
![Resources/[AI%20201]%20MINIPROJECT%20(23).png](Resources/[AI%20201]%20MINIPROJECT%20(23).png)
![Resources/[AI%20201]%20MINIPROJECT%20(24).png](Resources/[AI%20201]%20MINIPROJECT%20(24).png) -->


### Citation
If you find this work useful, please cite using the following:

```
@INPROCEEDINGS{10532443,
  author={Dela Cruz, Joshua R. and Miranda, Vince Raphael R. and Naval, Prospero C.},
  booktitle={2024 IEEE International Conference on Cybernetics and Innovations (ICCI)}, 
  title={Optimizing Rice Disease Classification: A Comparative Study of Different Color, Shape and Texture Features}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Support vector machines;Image segmentation;Histograms;Technological innovation;Image color analysis;Shape;Asia;computer vision;image classification;rice plant diseases;artificial neural network;support vector machine},
  doi={10.1109/ICCI60780.2024.10532443}}

```