# Hyperparameter Tuning in Causal ML Models: The Bayesian vs. Frequentist Approach

This repository contains the programmatic implementation of my (in progress) undergraduate thesis research for MMSS at Northwestern University. The main goal of this research is to compare the performance of Bayesian and frequentist hyperparameter tuning methods for X-learner models (from `EconML`) in the context of conditional average treatment effect (CATE) estimation.

## Overview


Machine learning models are extremely powerful predictive tools, and over the
last two decades, they have proliferated scientific and statistical applications.
The value of machine learning methods lies in both their quantitative capabilities and underlying structure: ML models can achieve incredible predictive
accuracy with high-dimensional data, and their flexibility allows them to adapt
in complex, nonlinear settings. In economics and related social science fields,
these capabilities have recently been integrated with classical methods of causal
inference, giving rise to the field of causal machine learning.


Causal ML combines machine learning algorithms with econometric identification strategies to estimate treatment effects; heterogeneous treatment effect
estimation is particularly valuable in observational settings where the efficacy
of intervention regimes benefits from personalization. Despite promising advancements in causal ML, empirical applications remain limited by challenges
that arise from both the causal side and the algorithmic sides, especially the
question of model configuration and tuning. In particular, hyperparameter tuning is frequently ad hoc or neglected in causal applications, undermining model
performance and broader empirical claims on treatment effects.


This thesis seeks to address the issue of hyperparameter tuning in the development of causal ML models, asking whether a Bayesian approach to hyper-
parameter tuning can deliver better performance than the standard frequentist-
style automatic tuning methods. The analysis focuses on causal ML methods
for estimating conditional average treatment effects, evaluating how such tuning
strategies affect the credibility of estimated treatment effects, and conceptualizing the problem of tuning from two opposing statistical philosophies.

