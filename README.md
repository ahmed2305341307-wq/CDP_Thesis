# CDP Thesis — Hierarchical Stacking

Full source and original dataset. Old model/result artifacts intentionally excluded.
Use Python 3.11 or 3.12 in a fresh virtual environment. From project root:

```
python -m pip install -r requirements.txt
python main.py
```

Alternatively run each file in main.py's order. The final phase asks for 15 already encoded/scaled feature values, not raw clinical measurements. Do not use as a clinical tool.

Architecture: RF, SVM, XGBoost -> level-1 logistic regression stack; RF/SVM/XGB probabilities plus level-1 probability -> level-2 gradient boosting (50 estimators, depth 2). Both stacks use stratified five-fold out-of-fold predictions. Original base model search ranges and feature counts retained. Meta learner parameters are fixed, not optimized. Soft voting and single-level stacking are retained only as comparison baselines. SHAP permutation explanation and LIME target the final stack. No guarantee of 90–94% accuracy. Original preprocessing/feature selection is fitted on the outer training split, but precedes inner CV and can bias inner CV estimates. Hyperparameter selection also precedes stacking CV. The reused test split is not an independent external validation set. For publication, implement fully nested evaluation with fold-local preprocessing/selection/tuning and independent validation. Report actual results.
