# Functions:
### 1) 
```python
lift_chart(chart_type = 'lift', scale = 100, cum = True, save = True)
```
chart_type: 
- 'lift'
- 'response'
- 'captured response'


### 2) 
```python
roc_chart(save = True)
```
AUC score

### 3) 
```python
score_hist_chart(density = True, cumulative = False, save = True)
```
density: if true then percentage scale
Kolmogorov-Smirnov test (D, p-value)

### 4) 
```python
precision_recall_chart(x_thresshold = False, save = True)
```
x_thresshold: if False then Precision Recall Curve
              if True then Precision Recall and Thresshold Curve
# Usage:
params - dictionary contains models names and sets. Pattern:
``` python
 params = {'model_name1' : (model1, {'set1_name' : (x1, y1), 'set2_name' : (x2, y2)}), 'model_name2' : ...}
```
```python
params = {}
params['cbc'] = (cbc_clf, {'train' : (train_prepared, train_y), 'test' : (test_prepared, test_y)})
model_ev = classifiereval(params = params)
model_ev.lift_chart(chart_type = 'lift', cum = True)
model_ev.roc_chart()
model_ev.score_hist_chart()
model_ev.precision_recall_chart(x_thresshold = True)
```

You can define a lot of models and sets in params. For each model graphs will be displayed. 
Models are comared between sets (in the future between models too).
# Results:
![alt text](https://github.com/mateusz-g94/classifiereval/blob/master/grp/cbc-lift.png)
![alt text](https://github.com/mateusz-g94/classifiereval/blob/master/grp/cbc-roc.png)
![alt text](https://github.com/mateusz-g94/classifiereval/blob/master/grp/cbc-train-score-hist.png)
![alt text](https://github.com/mateusz-g94/classifiereval/blob/master/grp/cbc-test-score-hist.png)
![alt text](https://github.com/mateusz-g94/classifiereval/blob/master/grp/cbc-train-PRTCurve.png)
![alt text](https://github.com/mateusz-g94/classifiereval/blob/master/grp/cbc-test-PRTCurve.png)
