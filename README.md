# Machine Failure Risk Evaluation

This is a small project that uses logistic regression to predict machine failures. The data is heavily imbalanced—failures are rare compared to normal runs. The goal is to produce a risk score that helps make better decisions, not just to maximize accuracy.

## The Problem

Machine failures do not happen often. Most of the time, machines run fine. This means the dataset has very few failure examples compared to normal ones. If you just try to maximize accuracy, you end up predicting "no failure" almost always—which is useless.

What we need is a probability score for each observation. Then we can decide: above what score do we treat it as high risk?

## Why Logistic Regression

I chose logistic regression for a few reasons:

- It gives a probability output directly, not just a class label.
- The coefficients are easy to read. You can see which features push the risk up or down.
- It is simple and works well when data is limited.

The model uses five features:
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

## What I Observed

I ran experiments at different decision thresholds. The ROC-AUC stayed at 0.897 across all runs. This makes sense once you think about it, because ROC-AUC is about ranking, not where you cut the line.

What changes is the recall and the false positive rate. Here are the results from the logs:

| Threshold | Recall | False Positive Rate | TP | FP | FN | TN |
|-----------|--------|---------------------|----|----|----|----|
| 0.416 | 0.718 | 0.055 | 28 | 108 | 11 | 1853 |
| 0.299 | 0.769 | 0.094 | 30 | 184 | 9 | 1777 |
| 0.165 | 0.821 | 0.192 | 32 | 376 | 7 | 1585 |
| 0.116 | 0.872 | 0.258 | 34 | 505 | 5 | 1456 |

After running these experiments, the pattern became pretty clear.

The threshold works like a **risk sensitivity slider**:
- Higher threshold (0.416) → fewer false alarms, but many failures missed.
- Lower threshold (0.116) → most failures caught, but many false alerts.

### Cost Framing

You can think of this as a cost problem:

```
Total Cost = (FN × Cost_A) + (FP × Cost_B)
```

- FN = missed failures (false negatives)
- FP = false alarms (false positives)
- Cost_A = what you lose when you miss a real failure (downtime, damage)
- Cost_B = what you lose on a false alarm (unnecessary inspection, wasted time)

There is no single "best" threshold. It depends on the actual costs in your situation.

## The Data Imbalance

The dataset has **9,661 normal runs** but only **339 failures**—roughly a 28:1 ratio. This is why accuracy is useless here. If you just predict "no failure" every time, you get 96.6% accuracy but catch zero failures.

![Machine Failure Frequency](images/Machine-failure-Frequency.png)

## Dashboard

The dashboard combines four key plots to help understand the model and choose a threshold:

![Dashboard](images/Dashboard_ai4i2020.png)

### What Each Plot Shows

**1. Confusion Matrix (top left)**  
At threshold 0.12, the model catches 34 out of 39 failures (TP=34, FN=5) but also flags 505 normal runs as risky (FP=505). This is the low-threshold scenario where recall is high but false alarms are many.

**2. ROC Curve (top right)**  
The **ROC-AUC is 0.897**. This tells you how well the model separates failures from normal runs. The score stays the same no matter what threshold you pick—you are just moving along the same curve.

**3. Precision-Recall Curve (bottom left)**  
Shows the trade-off between precision and recall. The **average precision is 0.412**. When you increase recall (catch more failures), precision drops (more false alarms). This plot is usually more useful than ROC when one class is rare.

**4. Risk Score Distribution (bottom right)**  
Shows how predicted risk scores are spread out. **Failures (blue) tend to cluster at higher scores**, while normal runs (green) are mostly near zero. But there is overlap, which is why the threshold matters. Some real failures have low scores (you will miss them if the threshold is high) and some normal runs have high scores (they become false alarms if the threshold is low).

These plots are decision aids. They do not tell you the "right" threshold—they show you the consequences of any threshold you pick.

## Logs

Each run writes a log file under `logs/`. The log includes the threshold used, data split, parameters, and evaluation metrics. The plots correspond to the most recent logged run, so the figures match the numbers in the logs.

I checked the results manually. I did not just accept whatever score the model gave.

## Failure and Limitation

An earlier version of this project failed and is preserved on the `Failure` branch. That attempt had a structural limitation that caused poor performance. The lesson from that failure influenced how the final model was designed.

---

The code here is intentionally minimal. It shows how to train, evaluate, and reason about trade-offs of a logistic regression risk scorer on imbalanced data.
