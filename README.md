# NeuroPy-MLToolbox

All the tools i created and used for neuroml during my phd

## Install instructions

```
pip install mlneurotools
```

## What's inside ?

in stats : ttest paired and unpaired with permutations, using parallel computing for the permutations. available corrections include FDR, maximum statistics, bonferonni.
Currently investigating if the FDR correction is working properly (might be something wrong with it according to some feedback, but was unable to recreate the bug)

in ml : A classification wrapper function that returns a dictionnary containing all the informations you need after your classification has been done. Warning: the classification function does not do either feature selection or hyperparameter serach, you will have to do it beforehand.
Also includes a StratifiedGroupShuffleSplit class, that is a stratified version of the GroupShuffleSplit from sklearn.

copyleft Arthur Dehgan
