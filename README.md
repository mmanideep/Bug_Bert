# Bug_Bert
Resource limitations frequently hinder the deployment of robust machine learning models in real-world applications. This research proposes BugBERT, a solution tailored for binary classification of software issues as either bugs or non-bugs, specifically within resource-constrained environments. Leveraging TinyBERT, a distilled variant of the BERT language representation model, BugBERT aims to achieve high classification accuracy while maintaining computational efficiency. Through this exploration, the study contributes to the field of efficient deep learning odels for specialized classification tasks on devices with limited resources.

## Repo Highlights
### Datasource
Json files containing the dataset of 5,591 tickets used in experiments. 
Dataset :scroll: is split into 7 files to avoid the anonymisation limitation of 1MB performed by _anonymous.4open.science_
The dataset coming from the conference paper: 
"[It’s not a bug, it’s a feature: how misclassification impacts bug prediction](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/05/icse2013-bugclassify.pdf)" 
by Herzig, Kim and Just, Sascha and Zeller, Andreas.)

### Repo overview
RaspberryPiUtils: Contains the scripts which were used to create docker image and run the API which wraps BugBert. Additionaly contains a sample script used to hit the API
BugBertTest.py: Script to test BugBert model
BugBertImplementation: Jupyter book to build BugBert
