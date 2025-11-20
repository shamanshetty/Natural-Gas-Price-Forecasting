# Natural Gas Price Forecasting with Hybrid LSTM + ARIMA Model

## Project Overview

This Jupyter notebook implements a hybrid forecasting model that combines LSTM (Long Short-Term Memory) neural networks with ARIMA (AutoRegressive Integrated Moving Average) to predict natural gas prices. The project was created by Shaman Shetty as part of a data science challenge.

## What We're Doing

The notebook performs the following key tasks:

1. **Data Loading & Cleaning**
   - Uploads a CSV file containing historical natural gas price data
   - Cleans and preprocesses the data, handling missing values and date formatting
   - Covers data from October 2020 to September 2024 (monthly intervals)

2. **Exploratory Data Analysis**
   - Visualizes the raw price data over time
   - Performs seasonal decomposition using STL (Seasonal and Trend decomposition using Loess)
   - Separates the time series into:
     - Trend component
     - Seasonal component
     - Residual component

3. **Hybrid Forecasting Architecture**
   - **LSTM Component**: Captures complex non-linear patterns and long-term dependencies in the data
   - **ARIMA Component**: Models linear trends and seasonal patterns
   - The hybrid approach combines the strengths of both methods for improved forecasting accuracy

## How It Works

### Data Preprocessing
```python
- Reads CSV with date and price columns
- Converts dates to datetime format
- Handles scientific notation in price values
- Removes any rows with missing data
- Sets date as index for time series analysis
```

### Seasonal Decomposition
The STL decomposition helps understand:
- **Trend**: Long-term directional movement in prices
- **Seasonality**: Regular cyclical patterns (12-month period for monthly data)
- **Residuals**: Random noise or irregular components

### Model Components
- **MinMaxScaler**: Normalizes price data for LSTM input
- **Sequential LSTM**: Deep learning model with LSTM layers for pattern recognition
- **ARIMA**: Statistical model for time series forecasting
- **Hybrid Integration**: Combines predictions from both models

## Requirements

```python
- pandas
- numpy
- matplotlib
- statsmodels (for ARIMA and STL)
- tensorflow/keras (for LSTM)
- scikit-learn (for preprocessing)
```

## Usage

1. Run the notebook in Google Colab (designed for Colab environment)
2. Upload your natural gas price CSV file when prompted
3. The CSV should have two columns:
   - Dates column
   - Prices column
4. The notebook will automatically:
   - Clean the data
   - Perform exploratory analysis
   - Train the hybrid model
   - Generate forecasts

## Data Format

Expected CSV structure:
```
Dates,Prices
2020-10-31,1.01E+01
2020-11-30,1.03E+01
...
```

## Key Features

- **Automatic data upload** via Google Colab file upload widget
- **Robust data cleaning** handles various date formats and scientific notation
- **Visual decomposition** of time series components
- **Hybrid modeling** approach for improved accuracy
- **Monthly forecasting** optimized for natural gas market analysis

## Notes

- The model uses a 12-month seasonal period appropriate for monthly data
- Data is automatically indexed by date for proper time series handling
- The notebook includes error handling for data loading and processing
- GPU acceleration is enabled in Colab for faster LSTM training
