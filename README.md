# Wikipedia-MRMR-Tag

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Data](#data)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Reproducibility](#reproducibility)
6. [Key Findings](#key-findings)
7. [References](#references)

---

## Problem Statement

**Objective:** his project is part of my thesis. During this thesis, I wanted to differenciate the style of the articles in Wikipedia by their categories, the problem is that there is no clear categories. The closest we have is some tags that are added by machine learning, such as 'geographical', 'history', but also 'Asia' or 'science & technologies'. 

Our objectif : Identifying a minimal set of topics that explained the best the stylistic variation in the articles.

**Motivation:** Wikipedia lacks clear, consistent categories. By leveraging machine-generated tags, we aim to uncover which topics most strongly predict stylistic differences, supporting better content organization and discovery.

---

## Data

- **Source:** 4,000 high-quality (featured/good) French Wikipedia articles (March 2022), scraped with PyWikibot.
- **Processing:** Each article version is vectorized using SpaCy (POS, morphological, syntactic tags).
- **Stylistic Clusters:** Neural Gas clustering (with Optuna hyperparameter tuning) yields 191 style clusters from ~1.4M article versions.

---

## Methodology

- **Regression Model:** ElasticNet regression predicts stylistic cluster density from topic tag densities.
- **Parameter Optimization:** Random search and 50-fold cross-validation maximize $R^2$.
- **Equations:**
  - $\hat{y} = \sum_C \beta_C N_{C,i}(t)$ where $\beta_C$ is a coefficient for the topic density, $N_{C,i}(t)$ is the number of article that have a style $i$ and a topic $C$.
  - $\epsilon = \sum_i (y_i - \hat{y}_i) + \alpha \text{penalty}$ is the equation that Elastic Net tries to minimize, with $\alpha$ being a parameter determining the importance of the penalty.
  - $\text{penalty} = \lambda ||beta||² + (1-\lambda) ||beta||_1$ where $\lambda$ is a hyperparameter and $||\beta||$ is the norm of the coefficient (wether L2 or L1). 

---

## Results


- **Best predictive model** : We achieve $R² > 0.85$ with a $\alpha < 10$ and this for any $\lambda \in [0,1]$ that we've tried
- **Top Predictive Tags:** With only 5 tags (STEM, Europe, Media, Asia, Biography) we succeed to explain most of the style variation.  
- **Interpretation:** Based on this, we can confidently consider only 5 topics as representative of the 4,000 articles of quality in Wikipedia. We see also that whether we used L1 or L2 norm was not very important, but the importance of the penalty was important.

---

## Key Findings

- Identified 5 tags explaining 95% of stylistic variation.

---

## Reproducibility

- All code in `main.ipynb`
- Data set available on [Kaggle](https://www.kaggle.com/datasets/jeanbaptistechaudron/stylistically-encoded-wikipedia-articles)
- Requirements: see `requirements.txt`

---

## References

- [ORES documentation](https://www.mediawiki.org/wiki/ORES/Articletopic#Taxonomy)
- [SpaCy](https://github.com/explosion/spacy-models/releases/tag/fr_core_news_lg-3.8.0)
- [Neural Gas Algorithm](link)
