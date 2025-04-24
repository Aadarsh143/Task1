# Task1
**Data Preprocessing Theory**
Preprocessing is the process of transforming raw data into a suitable format for further analysis or machine learning models. It improves data quality, ensures consistency, and enhances model performance. Preprocessing involves several steps, depending on the nature of the data and the intended use case.
Steps in Preprocessing
**1. Data Collection**
Gather raw data from various sources such as databases, APIs, CSV files, or IoT devices.
Ensure data completeness and reliability.
**2. Data Cleaning**
Handle missing values:
Remove rows/columns with missing data.
Impute missing values (mean, median, mode, interpolation).
Remove duplicate records to avoid redundancy.
Correct inconsistencies and errors (e.g., incorrect labels, typos).
**3. Data Transformation**
Convert data into a desired format for analysis.
Apply normalization or standardization:
Normalization (Min-Max Scaling): Scales values between 0 and 1.
Standardization (Z-score Scaling): Transforms data to have a mean of 0 and standard deviation of 1.
Encode categorical variables:
One-Hot Encoding: Converts categorical values into binary vectors.
Label Encoding: Assigns numerical labels to categories.
**4. Data Reduction**
Reduce dataset size without losing critical information.
Use techniques like:
Dimensionality Reduction (PCA, LDA)
Sampling (Random, Stratified)
Aggregation (Grouping data)
**5. Data Splitting**
Divide data into training, validation, and testing sets.
Common split ratios:
Training Set (70-80%): Used to train the model.
Validation Set (10-15%): Used for hyperparameter tuning.
Test Set (10-15%): Used for final model evaluation.
**6. Data Augmentation (For Image/Text Data)**
Apply transformations like rotation, flipping, cropping (for images).
Use synonym replacement, back translation (for text data).
