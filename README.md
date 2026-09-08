Repository containing the code used in the morphological linear enhancement experiment. It includes three main testing routines:

1. Length_Test.py --> performs morphological linear enhancement tests for different lengths of the linear structuring element B and generates plots of the evaluation metrics;
2. Linear_Enhacement.py --> performs morphological linear enhancement (MLE), including all stages defined in the proposed method, namely skeleton-based filtering, statistical filtering, and the final conditional dilation stage;
3. Skeleton_Threshold_Tests.py --> a series of tests to evaluate the effects of different thresholds on the filtering process based on skeleton analysis.

All routines point to the "Datasets" repository (https://github.com/dwfm77-bit/Dataset.git), which contains image samples for the different bands along with the revised masks. To access the complete dataset, use the link: https://www.kaggle.com/datasets/danielwander/s2-rd-lite. The routines can be executed locally. For dependencies, refer to the 'requirements.txt' file.
