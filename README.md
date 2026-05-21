# Right-Whale-Acoustics
Analysis and CNN-TCN Modeling of Right Whale Upcalls  
  
This is a repo to explore acoustic data of North Atlantic Right Whales acquired by NOAA in 2013.  The data was collected in the Stellwagen Bank National Marine Sanctuary east of Boston between Cape Ann and Cape Cod. 
The dataset and a detailed description may be found here: https://www.fisheries.noaa.gov/resource/data/noaa-nefsc-north-atlantic-right-whale-acoustic-data-and-annotations  
  
My primary interest here is to explore upshot calls of Right Whales and to build a CNN + TCN hybrid model capable of identifying such calls in an audio file.  
  
The dataset includes 8256 total events.  Additional datasets or a dateset augmented with synthetic data would be a prerequisite for a production-grade model which is highly generalizable to new ocean environments and different sensors, and able to detect rare call subtypes.  

Nonetheless, with minimal tuning this model gives an F1 score of 0.9217 (0.8849 precision, 0.9618 recall).  The ROC-AUC is 0.9900, and the PR-AUC is 0.9710.

I intend to fine tune this model soon on a second NOAA dataset inlcuding NEFSC data from multiple additional sources.  
  
Several similar models have been published recently.  
  
A recent example of a CNN model published by Hyer et al. utilizes multiple datasets and a system of augmentation to recognize upcalls of North Atlantic Right Whales.  
  
"Robust real-time detection of right whale upcalls using neural networks on the edge"  Ecological Informatics
Vol 89, 103130 (2025)  
https://doi.org/10.1016/j.ecoinf.2025.103130  

A recent example of a TCN model published by Goldwater et al. utilizes multiple datasets and synthetic data to recognize gunshot calls of North Pacific Right Whales.  
  
"Machine-learning-based simultaneous detection and ranging of impulsive baleen whale vocalizations using a single hydrophone" J. Acoust. Soc. Am. 153, 1094–1107 (2023)  
https://doi.org/10.1121/10.0017118  
  
  
![RightWhaleUpcallResults.png](https://github.com/aibatchelor22/Right-Whale-Acoustics/blob/main/RightWhaleUpcallResults.png)


To set up the package, clone the repo and run:  
pip install -e .  

To download the data and build the dataset:  
python scripts/download_data.py  
python scripts/build_dataset.py  

To start training:  
python scripts/train.py  

To evaluate the model:  
python scripts/evaluate.py <run directory>
