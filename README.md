# March Machine Learning Mania 2025

This repo is for my submission to the [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/overview) Kaggle competition.

This model shows improved performance over the winner of the 2023 competition, the last iteration with the 2025 scoring format.

This analysis creates a model based on team quality, seeding, and season statistics to predict the win probability of each game of the NCAA Men's and Women's tournament. The model uses an XGBoost algorithm with cross-validation to make predictions on Out-of-Bag games, then averages the results to create a predicted winning percentage (equalling a predicted win probability). A spline model is also overlayed to standardize the results and avoid overfitting.

## Competition Background

Per the competition overview:

> In our eleventh annual March Machine Learning Mania competition, Kagglers will once again join the millions of fans who attempt to predict the outcomes of this year's college basketball tournaments. Unlike most fans, you will pick the winners and losers using a combination of rich historical data and computing power, while the ground truth unfolds on television.
> You are provided data of historical NCAA games to forecast the outcomes of the Division 1 Men's and Women's basketball tournaments...You are making predictions about every possible matchup in the tournament, evaluated using the Brier score.
> Submissions are evaluated on the [Brier score](https://en.wikipedia.org/wiki/Brier_score) between the predicted probabilities and the actual game outcomes (this is equivalent to mean squared error in this context).

## Files

Analysis is split into 2 notebooks, and another notebook is used to format a final submission:

- [M_Analysis.ipynb](M_Analysis.ipynb) is the analysis notebook for the Men's tournament
- [W_Analysis.ipynb](W_Analysis.ipynb) is the analysis notebook for the Women's tournament
- [submission_formatting.ipynb](submission_formatting.ipynb) is a brief notebook which combines separate submission files from the Men's and Women's analyses into a final submission file
- The [submissions](submissions) folder contains the .csv files with intermediate and final predictions
- The [data](data) folder contains .csv files with competition data on Men's and Women's games. More information on the data available can be found on the competition [Data page](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data)

## Analysis

My algorithm and modelling process was heavily inspired by previous competition winners, in References below. My performance improvements over top solutions from previous years comes from differences in feature engineering and data pre-processing.

- Analysis was split into Men's and Women's notebooks separately.
  - This allows separate models to be trained on each dataset, resulting in different weights for features. E.g., the Women's predictions favor the higher seed even more than the Men's predictions.
  - This allows some additional data to be considered in feature engineering. E.g., the [GLM model](https://en.wikipedia.org/wiki/Generalized_linear_model) that estimates a team quality score considers data back to 2003 for the Men's dataset versus only 2010 for the Women's dataset.
- Additional features were calculated.
  - Original features included basic information like Points, Rebounds, and Assists.
  - 25 additional features were calculated, including Free Throw Percentage, True Shooting Percentage, Opponent Turnover Rate, and more.
  - The goal of the additional feature engineering was to standardize for pace of play, approximate opponent quality, and find feature interactions.

## Outcomes and Takeaways

Through 60 games of each tournament, this model slightly outperforms the competition winning model from 2023: 0.1717 versus 0.1737 Brier score. However, its performance in the 2025 competition has been average, and there are significant opportunities for improvement.

- Additional Features: the competition includes additional data that would improve the model.
  - Massey ordinals and other reputable computer team rankings
  - Conferences
  - Team Coaches
  - Performance in previous tournaments
- Additional features could also be accessed externally or calculated.
  - Betting lines
  - Elo model for team quality
- Model improvements could squeeze out slightly more performance.
  - Experimenting with other models, especially other tree-based algorithms like [LightGBM](https://lightgbm.readthedocs.io/en/stable/) and [CatBoost](https://catboost.ai/)
  - Ensembling multiple models with different algorithms
  - Hyperparameter tuning, likely with [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

Overall, as an exercise in feature engineering and process improvement, the improvements in this model over previous iterations is encouraging.

## References

- [paris madness 2023](https://www.kaggle.com/code/rustyb/paris-madness-2023/notebook)
- [paris madness](https://www.kaggle.com/code/raddar/paris-madness)
