from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# -------- Load model and encoders --------
def load_model():
    try:
        model_path = os.path.join(app.root_path, "price_forecast_model.pkl")
        encoders_path = os.path.join(app.root_path, "label_encoders.pkl")
        features_path = os.path.join(app.root_path, "feature_columns.pkl")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            label_encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}
            feature_cols = joblib.load(features_path) if os.path.exists(features_path) else []
            return model, label_encoders, feature_cols
        else:
            print("Model files not found - using fallback forecasting")
            return None, {}, []
    except Exception as e:
        print(f"Warning: Could not load model files: {e}")
        return None, {}, []

model, label_encoders, feature_cols = load_model()

# -------- Load dataset with robust date parsing --------
def load_data():
    try:
        csv_path = os.path.join(app.root_path, 'Agriculture_price_dataset.csv')
        if not os.path.exists(csv_path):
            print("Dataset file not found - creating sample data")
            return create_sample_data()
        
        df = pd.read_csv(csv_path)
        
        # Handle different date formats
        date_columns = ['Price Date', 'Date', 'price_date', 'date']
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['Price Date'] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        else:
            print("No date column found - adding sample dates")
            df['Price Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        df = df.dropna(subset=['Price Date'])
        
        # Standardize column names
        column_mapping = {
            'STATE': 'STATE',
            'State': 'STATE', 
            'state': 'STATE',
            'District Name': 'District Name',
            'District': 'District Name',
            'district': 'District Name',
            'Market Name': 'Market Name',
            'Market': 'Market Name',
            'market': 'Market Name',
            'Commodity': 'Commodity',
            'commodity': 'Commodity',
            'Modal_Price': 'Modal_Price',
            'Modal Price': 'Modal_Price',
            'Price': 'Modal_Price',
            'price': 'Modal_Price'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        
        # Ensure required columns exist
        required_cols = ['STATE', 'District Name', 'Market Name', 'Commodity', 'Modal_Price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing columns: {missing_cols} - creating sample data")
            return create_sample_data()
        
        # Clean price data
        df['Modal_Price'] = pd.to_numeric(df['Modal_Price'], errors='coerce')
        df = df.dropna(subset=['Modal_Price'])
        df = df[df['Modal_Price'] > 0]
        
        return df
        
    except Exception as e:
        print(f"Warning: Could not load dataset: {e} - creating sample data")
        return create_sample_data()

def create_sample_data():
    """Create sample data if the original dataset is not available"""
    np.random.seed(42)
    
    states = ['Maharashtra', 'Punjab', 'Uttar Pradesh', 'Gujarat', 'Rajasthan']
    districts = ['District A', 'District B', 'District C']
    markets = ['Market 1', 'Market 2', 'Market 3']
    commodities = ['Wheat', 'Rice', 'Onion', 'Potato', 'Tomato', 'Cotton']
    
    data = []
    start_date = datetime(2020, 1, 1)
    
    for i in range(1000):
        state = np.random.choice(states)
        district = np.random.choice(districts)
        market = np.random.choice(markets)
        commodity = np.random.choice(commodities)
        
        # Create realistic price variations
        base_prices = {'Wheat': 2000, 'Rice': 2500, 'Onion': 1500, 
                      'Potato': 1200, 'Tomato': 2000, 'Cotton': 5000}
        base_price = base_prices.get(commodity, 2000)
        
        # Add seasonal and random variations
        date = start_date + timedelta(days=np.random.randint(0, 1095))  # 3 years
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        price = base_price * seasonal_factor * (1 + np.random.normal(0, 0.2))
        price = max(price, base_price * 0.5)  # Minimum price floor
        
        data.append({
            'STATE': state,
            'District Name': district,
            'Market Name': market,
            'Commodity': commodity,
            'Modal_Price': round(price, 2),
            'Price Date': date
        })
    
    return pd.DataFrame(data)

df = load_data()

# -------- Enhanced Forecast function --------
def forecast_prices(state, district, market, commodity, months=6):
    try:
        # Filter historical data
        historical_data = df[
            (df['STATE'] == state) &
            (df['District Name'] == district) &
            (df['Market Name'] == market) &
            (df['Commodity'] == commodity)
        ].sort_values('Price Date')

        if historical_data.empty:
            return None, "No historical data found for the selected combination"

        # Get recent data for trend analysis
        recent_data = historical_data.tail(30)  # Last 30 records
        latest_price = recent_data['Modal_Price'].iloc[-1]
        
        # Calculate trend
        if len(recent_data) >= 2:
            prices = recent_data['Modal_Price'].values
            trend = np.polyfit(range(len(prices)), prices, 1)[0]
        else:
            trend = 0
        
        # Calculate seasonal patterns
        historical_data['Month'] = historical_data['Price Date'].dt.month
        monthly_factors = historical_data.groupby('Month')['Modal_Price'].mean()
        overall_mean = historical_data['Modal_Price'].mean()
        seasonal_factors = (monthly_factors / overall_mean).to_dict()
        
        # Generate forecasts
        current_date = datetime.now()
        forecasts = []
        
        for i in range(months):
            forecast_date = current_date + timedelta(days=30*i)  # Approximate month
            month = forecast_date.month
            year = forecast_date.year
            
            # Base forecast with trend
            base_forecast = latest_price + (trend * i)
            
            # Apply seasonal factor
            seasonal_factor = seasonal_factors.get(month, 1.0)
            
            # Add some realistic variation
            price_variation = 1 + np.random.normal(0, 0.05)  # 5% std deviation
            
            forecasted_price = base_forecast * seasonal_factor * price_variation
            forecasted_price = max(forecasted_price, latest_price * 0.5)  # Price floor
            
            forecasts.append({
                'Year': year,
                'Month': month,
                'Predicted_Price': round(forecasted_price, 2),
                'Date': forecast_date.strftime('%Y-%m-%d')
            })

        return forecasts, "Success"

    except Exception as e:
        return None, f"Error in forecasting: {str(e)}"

# -------- API Routes --------
@app.route('/api/districts/<state>')
def get_districts(state):
    if df.empty:
        return jsonify([])
    districts = sorted(df[df['STATE'] == state]['District Name'].dropna().unique().tolist())
    return jsonify(districts)

@app.route('/api/markets/<state>/<district>/<commodity>')
def get_markets(state, district, commodity):
    if df.empty:
        return jsonify([])
    markets = sorted(df[
        (df['STATE'] == state) &
        (df['District Name'] == district) &
        (df['Commodity'] == commodity)
    ]['Market Name'].dropna().unique().tolist())
    return jsonify(markets)

# -------- Main Routes --------
@app.route("/", methods=["GET", "POST"])
def index():
    if df.empty:
        return render_template("index.html", 
                             error="Dataset not found or could not be loaded.", 
                             states=[], districts=[], commodities=[], markets=[],
                             results=None, recommendation=None, comparison=None, 
                             plot_path=None, stats=None)

    # Get unique values for dropdowns
    states = sorted(df['STATE'].dropna().unique().tolist())
    commodities = sorted(df['Commodity'].dropna().unique().tolist())
    
    # Initialize variables
    districts, markets = [], []
    results = None
    recommendation = None
    comparison = None
    plot_path = None
    stats = None
    error = None

    # Get basic statistics
    stats = {
        'total_records': len(df),
        'unique_states': len(states),
        'unique_commodities': len(commodities),
        'date_range': f"{df['Price Date'].min().strftime('%Y-%m-%d')} to {df['Price Date'].max().strftime('%Y-%m-%d')}"
    }

    if request.method == "POST":
        selected_state = request.form.get("state")
        selected_district = request.form.get("district")
        selected_commodity = request.form.get("commodity")
        selected_market = request.form.get("market")
        forecast_months = int(request.form.get("months", 6))

        # Update districts based on selected state
        if selected_state:
            districts = sorted(df[df['STATE'] == selected_state]['District Name'].dropna().unique().tolist())
        
        # Update markets based on selections
        if selected_state and selected_district and selected_commodity:
            markets = sorted(df[
                (df['STATE'] == selected_state) &
                (df['District Name'] == selected_district) &
                (df['Commodity'] == selected_commodity)
            ]['Market Name'].dropna().unique().tolist())

        # Generate forecast if all fields are selected
        if all([selected_state, selected_district, selected_commodity, selected_market]):
            forecast, status = forecast_prices(
                selected_state, selected_district, selected_market, 
                selected_commodity, forecast_months
            )
            
            if status != "Success":
                error = status
            else:
                # Get historical data for plotting
                historical = df[
                    (df['STATE'] == selected_state) &
                    (df['District Name'] == selected_district) &
                    (df['Market Name'] == selected_market) &
                    (df['Commodity'] == selected_commodity)
                ].sort_values("Price Date")

                # Create enhanced plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Historical and forecast plot
                ax1.plot(historical['Price Date'], historical['Modal_Price'], 
                        'b-', linewidth=2, label='Historical Prices', marker='o', markersize=4)
                
                forecast_dates = [datetime.strptime(f['Date'], '%Y-%m-%d') for f in forecast]
                forecast_prices_list = [f['Predicted_Price'] for f in forecast]
                ax1.plot(forecast_dates, forecast_prices_list, 
                        'r--', linewidth=2, label='Forecasted Prices', marker='s', markersize=5)
                
                ax1.set_xlabel("Date", fontsize=12)
                ax1.set_ylabel("Price (₹)", fontsize=12)
                ax1.set_title(f"Price Forecast for {selected_commodity} in {selected_market}", fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
                
                # Price distribution histogram
                ax2.hist(historical['Modal_Price'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.axvline(historical['Modal_Price'].mean(), color='red', linestyle='--', 
                           label=f'Mean: ₹{historical["Modal_Price"].mean():.2f}')
                ax2.set_xlabel("Price (₹)", fontsize=12)
                ax2.set_ylabel("Frequency", fontsize=12)
                ax2.set_title("Historical Price Distribution", fontsize=14, fontweight='bold')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save plot
                static_dir = os.path.join(app.root_path, "static")
                if not os.path.exists(static_dir):
                    os.makedirs(static_dir)
                
                plot_filename = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plot_full_path = os.path.join(static_dir, plot_filename)
                plt.savefig(plot_full_path, dpi=300, bbox_inches="tight")
                plt.close()
                
                plot_path = f"static/{plot_filename}"

                # Prepare results
                results = [{
                    'Date': f['Date'],
                    'Predicted_Price': f['Predicted_Price']
                } for f in forecast]

                # Generate recommendation
                current_price = historical['Modal_Price'].iloc[-1]
                max_forecast = max(forecast, key=lambda x: x['Predicted_Price'])
                min_forecast = min(forecast, key=lambda x: x['Predicted_Price'])
                
                if max_forecast['Predicted_Price'] > current_price * 1.15:
                    recommendation = {
                        'action': 'Wait to Sell',
                        'reason': f"Prices expected to rise by {((max_forecast['Predicted_Price'] - current_price) / current_price * 100):.1f}%",
                        'best_time': max_forecast['Date'],
                        'best_price': max_forecast['Predicted_Price']
                    }
                elif min_forecast['Predicted_Price'] < current_price * 0.85:
                    recommendation = {
                        'action': 'Sell Now',
                        'reason': f"Prices expected to fall by {((current_price - min_forecast['Predicted_Price']) / current_price * 100):.1f}%",
                        'best_time': 'Immediately',
                        'best_price': current_price
                    }
                else:
                    recommendation = {
                        'action': 'Hold or Sell',
                        'reason': 'Prices expected to remain relatively stable',
                        'best_time': 'Flexible timing',
                        'best_price': current_price
                    }

                # Market comparison
                comparison_data = df[
                    (df['STATE'] == selected_state) &
                    (df['District Name'] == selected_district) &
                    (df['Commodity'] == selected_commodity)
                ]
                
                if not comparison_data.empty:
                    market_stats = comparison_data.groupby('Market Name')['Modal_Price'].agg(['mean', 'std', 'count']).round(2)
                    market_stats = market_stats.sort_values('mean', ascending=False)
                    comparison = market_stats.head(5).to_dict('index')

    return render_template(
        "index.html",
        states=states,
        districts=districts,
        commodities=commodities,
        markets=markets,
        results=results,
        recommendation=recommendation,
        comparison=comparison,
        plot_path=plot_path,
        error=error,
        stats=stats,
        selected_state=request.form.get("state", "") if request.method == "POST" else "",
        selected_district=request.form.get("district", "") if request.method == "POST" else "",
        selected_commodity=request.form.get("commodity", "") if request.method == "POST" else "",
        selected_market=request.form.get("market", "") if request.method == "POST" else ""
    )

@app.route("/analytics")
def analytics():
    if df.empty:
        return render_template("analytics.html", error="No data available")
    
    # Create analytics visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top commodities by average price
    top_commodities = df.groupby('Commodity')['Modal_Price'].mean().sort_values(ascending=False).head(10)
    top_commodities.plot(kind='bar', ax=ax1, color='lightcoral')
    ax1.set_title('Top 10 Commodities by Average Price', fontweight='bold')
    ax1.set_ylabel('Average Price (₹)')
    ax1.tick_params(axis='x', rotation=45)
    
    # State-wise price comparison
    state_prices = df.groupby('STATE')['Modal_Price'].mean().sort_values(ascending=False)
    state_prices.plot(kind='bar', ax=ax2, color='lightblue')
    ax2.set_title('Average Prices by State', fontweight='bold')
    ax2.set_ylabel('Average Price (₹)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Price trends over time
    monthly_trends = df.groupby(df['Price Date'].dt.to_period('M'))['Modal_Price'].mean()
    monthly_trends.plot(ax=ax3, color='green', linewidth=2)
    ax3.set_title('Price Trends Over Time', fontweight='bold')
    ax3.set_ylabel('Average Price (₹)')
    ax3.grid(True, alpha=0.3)
    
    # Price volatility by commodity
    volatility = df.groupby('Commodity')['Modal_Price'].std().sort_values(ascending=False).head(10)
    volatility.plot(kind='bar', ax=ax4, color='orange')
    ax4.set_title('Price Volatility by Commodity (Top 10)', fontweight='bold')
    ax4.set_ylabel('Price Standard Deviation')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save analytics plot
    static_dir = os.path.join(app.root_path, "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    analytics_filename = f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(os.path.join(static_dir, analytics_filename), dpi=300, bbox_inches="tight")
    plt.close()
    
    return render_template("analytics.html", 
                         plot_path=f"static/{analytics_filename}",
                         stats=stats if 'stats' in locals() else None)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)