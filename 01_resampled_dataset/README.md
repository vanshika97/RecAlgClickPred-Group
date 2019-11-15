## 1st Iteration of Synthetic Dataset.

### Shared the dataset through Google Drive due to size restrictions.

### Methods Applied before Resampling: 
> 1. Removed High NAN value features for each organisation_id
> 2. Filled Low NAN value features with mode of training set of that organisation_id’s dataset 
> (We can try to remove all NANs in the next iteration, since high number of synthetic values are produced, we can afford to lose bad data and try)
> 3. Removed Time data to enable Target Encoding
> 4. Target Encoded Categorical Data
> 5. Normalized (Standard Scaler) to Numerical Data
> 6. Steps 4 and 5 while preserving the indices to avoid NANs.

Below are the Original and Resampled Dataset shapes.

* Original Dataset Shape_01 Counter({0: 264700, 1: 5546})
* Resampled dataset Shape_01 Counter({1: 264700, 0: 264698})


* Original Dataset Shape_04 Counter({0: 99157, 1: 1058})
* Resampled dataset Shape_04 Counter({0: 99157, 1: 99157})


* Original Dataset Shape_08 Counter({0: 15067, 1: 159})
* Resampled dataset shape_08 Counter({0: 15067, 1: 15067})

