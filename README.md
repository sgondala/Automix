# Automix

This repository contains the code to use Fast AutoAugment algorithm to generate and pick automatic augmentations for a given task.

For the set of combinations, we used {4 EDA Augmentations, inter lada, intra lada, adverserial augmentation}

The way FAA Algorithm works is as follows -
- Create a model (M) on Train, and formulate a method to pick augmentations
- Repeat
  - Sample augmentations and created augmented val data 
  - Find loss of augmented val data on M
  - Save the augmentations with lowest loss (call this ‘aug val loss’)
- Best augmentations are the one with least loss
- Using saved augmentations, create the final model M’ on Train. 
- Use M’ to find out ‘final val accuracy’ and test accuracy

## Directions to run the code

yahoo_with_mixtext/train_*/ - Train a FAA policy using the chosen augmentation on the yahoo dataset.
yahoo_with_mixtext/evaluate_any_model.py - Once train generates a policy, run this to evaluate the policy
