# Activity Prediction for Chemical Compounds

This project was conducted between Annika Schavemaker, Anna Fernandez-Rajal and Anoogna Suresh Babu, as part of the course [ID2214 Programming for Data Science](https://www.kth.se/student/kurser/kurs/ID2214?l=en).

The goal is to find a suitable model to make predictions for the yet-unknown test labels. Herewith, it is of great importance that the chosen model does not only have good accuracy on the validation data, but also has a large area under the ROC curve (AUC-ROC).

### Processing the data

To generate features from SMILES, the open-source toolkit for cheminformatics RDKit is deployed. For the extraction of features within each atom, additional functions as described by Hirohara et al. were leveraged [1]. This included determining the number of hydrogen molecules, unsaturation, formal charges, total valence, ring structures, aromatics, chirality, and hybridization. For this approach, features had to be extracted from each atom within each SMILE, resulting in a
large time to process the data.

The preprocessed data were split into a training set (75%) and a validation set (25%), after which a column filter was used to define labels and features. The label the model aims to predict is ’ACTIVE. 

Different representations were applied, and the data was normalized with a MinMax Scaler followed by a Standard Scaler and principle component analysis (PCA). This step was performed to reduce a large number of features in the data, with minimal reduction of information. Another approach was tried using only a RobustScaler before applying PCA and after the splitting of the initial data. This aimed to remove the outliers that for our selected features didn’t show relevant changes.

### Conclusion
From the results, we were able to observe that the accuracies for all models are very high ranging from 98% and 99%. Nevertheless, there are slight differences in the model performance with respect to the area under the ROC curve. The highest AUC score on the validation set, being 77.8%, is obtained using an artificial neural network with the parameters solver = adam and alpha = 0.0001. Therefore, the estimate of the AUC on the test set is approximately 77%.

## References
<a id="1">[1]</a> 
Hirohara, M., Saito, Y., Koda, Y. et al. Convolutional neural network based on SMILES representation of compounds for detecting chemical motif. BMC Bioinformatics 19 (Suppl 19), 526 (2018). https://doi.org/10.1186/s12859-018-2523-5
