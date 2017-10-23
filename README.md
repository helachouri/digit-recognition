# Digit Recognition
Handwritten Digit Recognition using TensorFlow, Keras and Python 

# Dependencies
1. `tensorflow`
2. `keras`
3. `numpy`
4. `PIL`
5. `collections`

# Contents
This repository contains the following files :

1. `classifier_generator.py` - Python Script to create the classifier files
2. `perform_recognition.py` - Python Script to test the classifier
3. `model.h5` - Classifier weights file for digit recognition
4. `model.json` - Classifier model file for digit recognition
5. `Desktop/Deep_learning/numbers` - Test images folder
    
# Usage
    
* Clone the repository :
```bash
git clone https://github.com/Hamza-El-Achouri/digit-recognition.git
cd digit-recognition
```
* The next step is to train the classifier. To do so run the script `classifier_generator`. It will produce the classifier files named `model.h5` and `model.json`. 

**NOTE** - *I have already created the `model.h5` and `model.json`, so this step is not necessary.*
```python
python classifier_generator.py
```
* To test the classifier, run the `perform_recognition.py` script.
```python
python3 perform_recognition.py -i <path to test image>
```
ex :
```python
python perform_recognition.py -i Desktop/Deep_learning/numbers/0
```
     
