# NLP - Binary Classification - Mini project

**Goal**: 

Classify each word to one of 2 classes; True / False. Specifically here - whether the word has been tagged or not.

**Data**: 

train.tagged - imbalanced dramatically, dev.tagged, test.untagged

**Tools**:

pytorch, scikit-learn, GloVE embedding

**Models**:
 - Classy model - SVM - using GridSearch for hyperparameters.
   - model's code: [first model code](HW1/first_model.py)
   - pre trained model:  [first model pickle](HW1/first_model_glove25_no_balance.pickle)
 - Simple CNN - fully connected with activation, dropuout and batch normalization
   - model's code: [second model code](HW1/second_model.py)
   - pre trained model - weights:  [second model pt](HW1/second_model.pt)

**Results**:

tagged test files based on trained models' predictions.(comp_mx_.tagged files).

f1 score + confusion matrix.

![confusion_mat_glove25_no_balance](https://user-images.githubusercontent.com/62693687/146646376-41a5cbb8-226f-4d74-9569-f40651a087aa.png)



--------------------------------------------------------------------------------------

Additional info can be found in the [report](HW1/report_313177412.pdf).

*This mini project was created in the scope of NLP course HW in the Technion.
