# Right-Whale-Acoustics
Analysis and TCN Modeling of Right Whale Upcalls  
  
This is a repo to explore acoustic data of North Atlantic Right Whales acquired by NOAA in 2013.  The data was collected in the Stellwagen Bank National Marine Sanctuary east of Boston between Cape Ann and Cape Cod. 
The dataset and a detailed description may be found here: https://www.fisheries.noaa.gov/resource/data/noaa-nefsc-north-atlantic-right-whale-acoustic-data-and-annotations  
  
My primary interest here is to explore upshot calls of Right Whales and to build a CNN + TCN hybrid model capable of identifying such calls in an audio file.  
  
This is a preliminary study and I do not expect to generate a highly robust production-grade model.  I do not expect it to be highly generalizable to new ocean environments, different sensors, or to be able to detect rare call subtypes.  The dataset includes 529 total events.  Additional datasets or a dateset augmented with synthetic data would be a prerequisite for a production-grade model.  

Nonetheless, with minimal tuning this model gives an F1 score of 0.9217 (0.8849 precision, 0.9618 recall).  
  
A recent example of a CNN model published by Hyer et al. utilizes multiple datasets and a system of augmentation to recognize upcalls of North Atlantic Right Whales.  
  
"Robust real-time detection of right whale upcalls using neural networks on the edge"  Ecological Informatics
Vol 89, 103130 (2025)  
https://doi.org/10.1016/j.ecoinf.2025.103130  

A recent example of a TCN model published by Goldwater et al. utilizes multiple datasets and synthetic data to recognize gunshot calls of North Pacific Right Whales.  
  
"Machine-learning-based simultaneous detection and ranging of impulsive baleen whale vocalizations using a single hydrophone" J. Acoust. Soc. Am. 153, 1094–1107 (2023)  
https://doi.org/10.1121/10.0017118  
  
!([RightWhaleUpcallResults.png](https://github.com/aibatchelor22/Right-Whale-Acoustics/blob/main/RightWhaleUpcallResults.png))
