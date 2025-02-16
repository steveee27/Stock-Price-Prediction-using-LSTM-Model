# **Stock Price Prediction using LSTM Architecture**

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview
This project focuses on predicting stock prices of **Apple (AAPL)** and **AMD (AMD)** using **LSTM (Long Short-Term Memory)** neural networks. The goal is to forecast future stock prices based on historical stock data. The project follows a step-by-step approach, from data exploration and preprocessing to building and evaluating multiple LSTM models. Modifications like adding dropout, regularization, and fine-tuning hyperparameters are explored to improve the model performance.

## Dataset Overview
The datasets for this project contain historical stock data for **Apple** (AAPL) and **Advanced Micro Devices (AMD)**:
- **Apple (AAPL)**: The dataset includes daily stock prices from 1980 to 2020.
- **AMD (AMD)**: The dataset includes daily stock prices from 1980 to 2020.

Each dataset consists of columns such as `Date`, `Open`, `High`, `Low`, `Close`, `Adjusted Close`, and `Volume`.

## Data Preprocessing
Data preprocessing steps include:
1. **Rescaling**: Data is normalized using **Min-Max scaling** to bring all values into the range [0, 1], which helps improve model convergence.
2. **Windowing**: The time series data is split using a window size of 5, meaning the model will use data from 5 days to predict the price on the 6th day (next day).
3. **Splitting the Dataset**: The dataset is split into:
   - **80%** training data
   - **10%** validation data
   - **10%** test data

The datasets for both **AAPL** and **AMD** are prepared separately for training and evaluation.

## Model Architecture
Two models are implemented for stock price prediction:
1. **Baseline Model**: The first model is a simple **LSTM** architecture with 50 units and one output layer. It uses ReLU activation for the LSTM layer and Adam optimizer.
2. **Modified Model**: The second model enhances the baseline by adding a second LSTM layer, dropout regularization (with a dropout rate of 0.2), L2 regularization, and tuning hyperparameters such as learning rate and epochs.

### Model Parameters:
- **LSTM Layers**: The first LSTM layer has 50 units, and the second LSTM layer has 30 units.
- **Dropout**: Dropout layers are added to reduce overfitting.
- **L2 Regularization**: Used on both LSTM and Dense layers to prevent overfitting.

## Model Training and Evaluation
Training is carried out using **Adam optimizer**, and the models are evaluated on the **test set** using metrics like:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**

### Evaluation Results:
- **Baseline Model**: Trained for 50 epochs, but there was some overfitting, especially with volatile stock price data.
- **Modified Model**: Achieved better performance with improved generalization and stability due to the additional LSTM layer, dropout, and regularization techniques.

## Results and Discussion

### Model Performance
- **Baseline Model**: The baseline model for **AAPL** and **AMD** had a relatively good performance in following general trends but faced issues like overfitting, especially during periods of high volatility.
- **Modified Model**: The second model with modifications like dropout and additional LSTM layers performed better. The **validation loss** became more stable, and the model managed to capture more complex patterns in stock data.

### Key Insights
1. **Model Stability**: The modified model improved stability, showing less fluctuation in validation loss compared to the baseline model.
2. **Overfitting**: Both models showed overfitting to some extent, especially during volatile periods, which indicates the need for more regularization.
3. **Stock Price Volatility**: Despite the improvements, the models faced difficulty in predicting extreme price changes, a common issue in financial markets.

### Limitations
- **Overfitting**: While the modified model improved performance, overfitting remained a challenge. Regularization techniques like L2 helped, but further improvements could be made with more data or advanced techniques.
- **Stock Market Volatility**: The models struggled with periods of high volatility, where stock prices can change unpredictably, making predictions more difficult.

## Conclusion
This project demonstrates the potential of **LSTM networks** in predicting stock prices, especially for data with temporal patterns. While both the **baseline model** and the **modified model** performed well in capturing general trends, the **modified model** proved to be more effective in generalizing to unseen data. Future work could involve exploring other architectures like **GRU**, increasing data volume, and applying more sophisticated techniques to handle volatility in the stock market.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
