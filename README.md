MLOps Assignment №1 at MADE
==============================
Prediction of heart disease with various classifiers (https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
# Usage example:
Train:
~~~
python3 core/train.py configs/train_config.yaml
~~~
Inference:
~~~
python3 core/predict.py models/model.pkl data/raw/heart_cleveland_test.csv data/raw/preds.csv
~~~
# Available classifiers:
<ul>
<li>LogisticRegression</li>
<li>Linear SVM</li>
<li>RBF SVM</li>
<li>Gaussian Process</li>
<li>Decision Tree</li>
<li>Random Forest</li>
<li>Neural Net</li>
<li>AdaBoost</li>
<li>QDA</li>
<li>CatBoost</li>

</ul>	
