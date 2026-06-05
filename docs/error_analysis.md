# Error Analysis

This document describes the model error patterns to inspect after running:

```bash
python scripts/download_dataset.py --dataset dreaddit
python notebooks/train_pipeline.py
```

The pipeline exports the leaderboard, confusion matrix, and feature attribution artifacts used for analysis.

## Current Audit Summary

The repository has a strong classical NLP pipeline: preprocessing, TF-IDF, Word2Vec comparison, sentiment features, emotion lexicon features, linguistic pattern features, class imbalance handling, model benchmarking, cross-validation, explainability, and Streamlit deployment.

The main credibility weakness is dataset quality. The original synthetic generator is useful for demonstrating mechanics, but resume and Kaggle value improve substantially when results are generated from a public dataset such as Dreaddit.

## Common False Positives

False positives are expected when posts contain strong negative emotion but do not describe sustained stress or mental health risk. Typical examples include:

- Temporary frustration about work, exams, or relationships
- Sarcasm or exaggerated language such as "I hate everything today"
- High punctuation or capitalization that resembles distress signals
- Posts with negative sentiment but no persistent self-focused distress pattern

These errors suggest that the model may overweight surface-level negative vocabulary when context is limited.

## Common False Negatives

False negatives are expected when distress is indirect, short, or phrased without obvious emotional keywords. Typical examples include:

- Understated posts such as "I am tired of doing this every day"
- Humor used to mask emotional distress
- Posts describing avoidance or withdrawal without explicit sadness or fear words
- Text with neutral sentiment but concerning context

These errors are important because recall matters in sensitive-domain classification. They indicate that lexical features may miss pragmatic context.

## Common Confusion Patterns

The most important confusion patterns are:

- Stress versus normal negative emotion
- Anxiety-like worry versus general stress
- Sadness-related vocabulary versus broader mental health signal labels
- Short posts where there is not enough context for the model
- Mixed emotional states where one post contains fear, sadness, anger, and fatigue

For Dreaddit, the label space is binary, so the primary confusion is between `normal` and `stress_signal`. For the synthetic three-class fallback, likely confusion occurs between `depression` and `anxiety` because both can contain negative sentiment, first-person language, and high emotional intensity.

## Key Observations

Classical linear models are expected to perform well with TF-IDF because the feature space is sparse and high-dimensional. Tree-based models may be less competitive on sparse lexical features but can still help inspect engineered features.

SMOTE can improve minority-class learning, but it should be interpreted carefully for text-derived feature spaces because synthetic feature vectors may not correspond to natural examples.

SHAP and coefficient analysis should be reviewed for spurious correlations. In a credible mental health NLP project, it is not enough to report high metrics; the important question is whether the model relies on meaningful language patterns rather than dataset artifacts.

## Recommended Manual Review Workflow

After training, export misclassified examples from the test set and manually review 20 to 30 examples:

- 10 false positives
- 10 false negatives
- 5 to 10 high-confidence errors

For each example, record:

- True label
- Predicted label
- Prediction confidence if available
- Top TF-IDF or SHAP features
- Human interpretation of why the model may have failed

This review is valuable for GitHub, Kaggle, and interview discussion because it shows model evaluation maturity beyond leaderboard scores.
