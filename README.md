# Agriculture Price Forecasting System

A modern web application built with Flask for predicting agriculture commodity prices using machine learning and statistical analysis.

## Features

### 🚀 Core Functionality
- **Price Forecasting**: Predict commodity prices for 3, 6, or 12 months
- **Dynamic Filtering**: State → District → Market → Commodity selection
- **Interactive Visualizations**: Historical trends and forecasting charts
- **Market Comparison**: Compare prices across different markets
- **Smart Recommendations**: AI-powered buying/selling suggestions
- **Analytics Dashboard**: Comprehensive market analytics and insights

### 🎨 Modern UI/UX
- **Responsive Design**: Works on all devices (mobile, tablet, desktop)
- **Modern Styling**: Bootstrap 5 with custom CSS and animations
- **Interactive Elements**: Dynamic dropdowns, hover effects, loading spinners
- **Professional Charts**: High-quality matplotlib visualizations
- **Intuitive Navigation**: Clean and user-friendly interface

### 📊 Advanced Analytics
- Top commodities by average price
- State-wise price comparisons
- Historical price trends
- Price volatility analysis
- Seasonal pattern detection

## Project Structure

```
agriculture-price-forecasting/
│
├── app.py                      # Main Flask application
├── templates/
│   ├── index.html             # Main forecasting page
│   └── analytics.html         # Analytics dashboard
├── static/                    # Auto-generated charts and assets
├── requirements.txt           # Python dependencies
├── Agriculture_price_dataset.csv  # Your dataset (optional)
├── price_forecast_model.pkl   # Trained model (optional)
├── label_encoders.pkl         # Label encoders (optional)
├── feature_columns.pkl        # Feature columns (optional)
└── README.md                  # This file
```

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### 1. Clone or Download
Download the project files and organize them according to the project structure above.

### 2. Install Dependencies
```bash
pip install flask pandas numpy joblib matplotlib seaborn scikit-learn
```

Or create a requirements.txt file:
```
Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
Werkzeug==2.3.7
```

Then install:
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data (Optional)
- Place your CSV file as `Agriculture_price_dataset.csv` in the root directory
- The app will create sample data if no dataset is found
- Expected columns: `STATE`, `District Name`, `Market Name`, `Commodity`, `Modal_Price`, `Price Date`

### 4. Run the Application
```bash
python app.py
```

The application will start at `http://localhost:5000`

## Usage Guide

### 1. Main Forecasting
1. **Select Parameters**: Choose State → District → Commodity → Market
2. **Set Forecast Period**: Choose 3, 6, or 12 months
3. **Generate Forecast**: Click the "Generate Forecast" button
4. **View Results**: 
   - Forecast table with predicted prices and trends
   - Interactive price visualization charts
   - Smart buying/selling recommendations
   - Market comparison data

### 2. Analytics Dashboard
1. **Navigate to Analytics**: Click "Analytics" in the navigation menu
2. **View Comprehensive Charts**: 
   - Top commodities by price
   - State-wise price comparisons
   - Historical price trends
   - Price volatility analysis
3. **Use Insights**: Apply the analytics for strategic decisions

## API Endpoints

### Internal APIs (for dynamic dropdowns)
- `GET /api/districts/<state>` - Get districts for a state
- `GET /api/markets/<state>/<district>/<commodity>` - Get markets for selection

## Data Requirements

### CSV Format
Your agriculture dataset should have these columns:
```
STATE, District Name, Market Name, Commodity, Modal_Price, Price Date
```

### Sample Data Structure
```csv
STATE,District Name,Market Name,Commodity,Modal_Price,Price Date
Maharashtra,Pune,Pune Market,Onion,2500.00,2023-01-15
Punjab,Ludhiana,Grain Market,Wheat,2200.00,2023-01-15
```

### Data Quality Tips
- Ensure dates are in recognizable format (YYYY-MM-DD, DD/MM/YYYY, etc.)
- Remove or handle missing values
- Ensure price data is numeric and positive
- Include sufficient historical data for better predictions

## Forecasting Algorithm

### Default Method (Without ML Model)
1. **Trend Analysis**: Calculate recent price trends using linear regression
2. **Seasonal Factors**: Analyze monthly price patterns from historical data
3. **Price Variation**: Apply realistic market fluctuations
4. **Constraints**: Apply price floors to prevent unrealistic predictions

### With ML Model (Optional)
- Load pre-trained joblib models: `price_forecast_model.pkl`
- Use label encoders: `label_encoders.pkl`
- Feature engineering: `feature_columns.pkl`

## Customization Options

### 1. Styling & Themes
Modify CSS variables in the templates:
```css
:root {
    --primary-color: #2c5530;    /* Main green theme */
    --secondary-color: #4a7c59;  /* Secondary green */
    --accent-color: #7fb069;     /* Accent color */
}
```

### 2. Forecasting Parameters
In `app.py`, modify the `forecast_prices` function:
```python
# Adjust seasonal factors
if month in [6, 7, 8]:  # Monsoon months
    seasonal_factor = 1.2
elif month in [11, 12, 1]:  # Winter months
    seasonal_factor = 0.9
```

### 3. Chart Styling
Modify matplotlib settings:
```python
plt.style.use('seaborn-v0_8')  # Change chart style
sns.set_palette("husl")        # Change color palette
```

## Features Explained

### 🔮 Smart Forecasting
- **Trend-based Predictions**: Uses recent price movements
- **Seasonal Adjustments**: Accounts for agricultural seasons
- **Market Reality**: Applies realistic price constraints
- **Multiple Timeframes**: 3, 6, or 12-month predictions

### 📈 Visual Analytics
- **Historical vs Predicted**: Side-by-side comparison
- **Price Distribution**: Histogram of historical prices
- **Trend Indicators**: Rising/falling price indicators
- **Market Comparison**: Best markets for each commodity

### 💡 Intelligent Recommendations
- **Optimal Timing**: When to buy or sell
- **Price Expectations**: Expected price changes
- **Risk Assessment**: Market volatility indicators
- **Action Plans**: Clear recommendations with reasoning

### 🌐 Responsive Design
- **Mobile-First**: Optimized for all screen sizes
- **Touch-Friendly**: Large buttons and touch targets
- **Fast Loading**: Optimized images and assets
- **Accessibility**: Proper contrast and semantic markup

## Troubleshooting

### Common Issues

1. **Dataset Not Loading**
   ```
   Solution: Check CSV file path and column names
   Fallback: App creates sample data automatically
   ```

2. **Empty Dropdowns**
   ```
   Solution: Ensure data has the required columns
   Check: STATE, District Name, Market Name, Commodity columns
   ```

3. **No Forecast Generated**
   ```
   Solution: Ensure all dropdowns are selected
   Check: Network connectivity for AJAX calls
   ```

4. **Charts Not Displaying**
   ```
   Solution: Check if static/ directory is created
   Ensure: Matplotlib backend is properly configured
   ```

### Performance Optimization

1. **Large Datasets**
   ```python
   # Add data filtering to improve performance
   df = df.tail(10000)  # Use recent 10,000 records only
   ```

2. **Chart Generation**
   ```python
   # Reduce chart DPI for faster generation
   plt.savefig(plot_path, dpi=150, bbox_inches="tight")
   ```

## Deployment

### Local Development
```bash
python app.py
# Access: http://localhost:5000
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
# Create Dockerfile and docker-compose.yml as needed
```

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Comment complex logic
- Test all features before committing

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn (optional)
- **Styling**: Custom CSS with CSS Grid and Flexbox

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with sample data
4. Create detailed issue reports

## Future Enhancements

-  Real-time data integration
-  Advanced ML models (LSTM, ARIMA)
-  User authentication and profiles
-  Historical forecast accuracy tracking
-  API for third-party integrations
-  Mobile app development
-  Multi-language support
-  Advanced filtering options
-  Export functionality (PDF, Excel)
-  Email alerts for price changes

**Built with ❤️ for farmers and agricultural businesses**

Price_forecast_model.pkl file can be found here: https://drive.google.com/file/d/16rsqxOHojliwx_RahhwiKM0rYkhxwlcu/view?usp=sharing

Dataset link can be found here: https://drive.google.com/file/d/1ugpS0h22Zsv2-EtC0W1CM9pVneNLZYKF/view?usp=sharing

Dataset taken from kaggle: Indian Agricultural Mandi Prices (2023–2025)
