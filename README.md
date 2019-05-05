# ERP PREDICTION CONTEST
The UCSB Department of Statistics &amp; Applied Probability and the Center for Financial Mathematics and Actuarial Research are partnering with Hull Tactical to run a data science contest on predicting the S&P 500 returns. 

Link to competition: https://ucsb-erp-contest.herokuapp.com/

## Tools and Methods Used In Project
- Pytorch
- Fastai
- Multiple and Single Input Convolutional Neural Networks
- Image Creation using Gramian Angular Fields

## Aim
Images are created using Gramian Angular Field [1] of time series data. Using Python, Fastai and Pytorch a Convolutional Neural Network is created to take multiple image imputs.

[1] Image representation created using Gramian Angular Field described by Zhiguang Wang and Tim Oates (Jun 2015) here: https://arxiv.org/pdf/1506.00327.pdf

## Description of files:
- `4 percentage+BB+MA20050diff+MACD.html`: The same notebook as `4 percentage+BB+MA20050diff+MACD.ipynb` where all steps has bee commented.
- `dataset.csv`: The data for the competition, can be found at https://ucsb-erp-contest.herokuapp.com/
- `imagecreater.py`: Adds relevant features to the data set and creates image representation of time series data
- `ResNet50 CLOSE.ipynb`: Creates, trains and tests a CNN for single input: CLOSE.
- `ResNet50 percentage.ipynb`: Creates, trains and tests a CNN for single input: percentage.
- `2 CLOSE+MA20050diff.ipynb`: Creates, trains and tests a CNN for 2 inputs: CLOSE and MA20050diff.
- `2 percentage+BB.ipynb`: Creates, trains and tests a CNN for 2 inputs: percentage and BB.
- `2 percentage+RSI.ipynb`: Creates, trains and tests a CNN for 2 inputs: percentage and RSI.
- `4 CLOSE+percentage+RSI+BBdiff.ipynb`: Creates, trains and tests a CNN for 4 inputs: CLOSE, percentage, RSI and BB.
- `4 percentage+BB+MA20050diff+MACD.ipynb`: Creates, trains and tests a CNN for 4 inputs: percentage, BB, MA20050diff, MACD. __All steps has bee commented.__

## Note
The file containing the images is too large to upload. The images can be reconstructed using the `imagecreater.py` and `dataset.csv`.
