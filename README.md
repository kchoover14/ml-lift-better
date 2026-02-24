## Form Matters--Your Wearables Know

Fitness tracking typically measures whether people exercise, not how well they move. This project applies four machine learning classifiers to accelerometer data from sensors worn on the belt, forearm, upper arm, and dumbbell during supervised bicep curl sessions, comparing their ability to distinguish correct form from four specific technique errors. The random forest model achieved 99.4% accuracy across all five form categories -- a result that held for all five error classes and came with an out-of-sample error of just 0.6%. Model accuracy peaked at 27 of 52 predictors, indicating that a leaner deployment model is achievable with targeted feature selection.

## Portfolio Page

The [portfolio page](https://kchoover14.github.io/ml-lift-better) includes a full project narrative, key findings, and figures.

## Tools & Technologies

**Languages:** R

**Tools:** caret | renv

**Packages:** dplyr | ggplot2 | GGally | caret | rattle | party | rpart.plot | randomForest | gbm

## Expertise

Demonstrates ability to select and compare competing model types, diagnose failure modes in underperforming algorithms, and translate classification results into actionable deployment recommendations for applied technology contexts.

## License

- Code and scripts are licensed under the [MIT License](LICENSE).
- Data, figures, and written content © Kara C. Hoover, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
