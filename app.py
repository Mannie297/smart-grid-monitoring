"""
Smart Grid Monitoring Dashboard
-----------------------------
A comprehensive application for monitoring and analyzing smart grid performance using 
machine learning and interactive visualizations.

Main Features:
- Real-time monitoring of grid metrics
- Load prediction using multiple ML models
- Feature importance analysis
- Interactive data visualization
- Energy source distribution analysis

Author: [Emmanuel.O]
Date: [2025-04-30]
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import os

# Page Configuration
# -----------------
st.set_page_config(
    page_title="Smart Grid Monitoring",
    page_icon="⚡",
    layout="wide"
)

# Application Title
st.title("Smart Grid Monitoring Dashboard")

# Initialize Session State Variables
# --------------------------------
# These variables persist across reruns of the app
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_scaler' not in st.session_state:
    st.session_state.feature_scaler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None

@st.cache_data
def load_data():
    """
    Load and preprocess the smart grid dataset.
    
    Returns:
        pandas.DataFrame: Processed dataset with standardized column names and additional calculated features.
        
    Notes:
        - Handles missing data
        - Standardizes column names
        - Calculates derived features
        - Adds synthetic data where needed
    """
    try:
        # Attempt to load data from different possible locations
        file_path = '../smart_grid_dataset.csv'
        if not os.path.exists(file_path):
            file_path = 'smart_grid_dataset.csv'
            if not os.path.exists(file_path):
                st.error(f"Data file not found. Tried paths: {os.path.abspath('../smart_grid_dataset.csv')} and {os.path.abspath('smart_grid_dataset.csv')}")
                return None
            
        # Load and preprocess the dataset
        df = pd.read_csv(file_path)
        
        # Standardize timestamp handling
        if 'Timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.drop('Timestamp', axis=1)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(r'\s*\([^)]*\)', '', regex=True)
        df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
        
        # Standardize column names for consistency
        column_mapping = {
            'power_consumption': 'load',
            'solar_power': 'solar_generation',
            'wind_power': 'wind_generation',
            'grid_supply': 'grid_generation',
            'voltage_fluctuation': 'grid_stability',
            'electricity_price': 'price_per_kwh',
            'power_factor': 'power_factor',
            'temperature': 'temperature',
            'humidity': 'humidity'
        }
        
        # Only rename columns that exist
        column_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=column_mapping)
        
        # Add synthetic data for missing columns
        if 'natural_gas_generation' not in df.columns:
            df['natural_gas_generation'] = df['grid_generation'] * 0.4
        if 'coal_generation' not in df.columns:
            df['coal_generation'] = df['grid_generation'] * 0.3
        if 'nuclear_generation' not in df.columns:
            df['nuclear_generation'] = df['grid_generation'] * 0.3
            
        # Add voltage and current if not present
        if 'voltage' not in df.columns:
            df['voltage'] = 230 + np.random.normal(0, 2, len(df))
            
        if 'current' not in df.columns:
            df['current'] = (df['load'] * 1e6) / df['voltage']
            
        if 'price_per_kwh' not in df.columns:
            base_price = 0.15
            time_factor = np.sin(2 * np.pi * (df['timestamp'].dt.hour / 24)) * 0.03
            demand_factor = (df['load'] - df['load'].mean()) / df['load'].std() * 0.02
            df['price_per_kwh'] = base_price + time_factor + demand_factor
            df['price_per_kwh'] = df['price_per_kwh'].clip(0.10, 0.20)
        
        # Calculate additional metrics
        df['grid_losses'] = df['load'] * np.random.uniform(0.03, 0.05, len(df))
        df['voltage_stability'] = 100 - df['grid_stability']
        
        if 'frequency' not in df.columns:
            df['frequency'] = np.random.normal(50, 0.1, len(df))
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Error details:", e.__class__.__name__)
        if 'df' in locals():
            st.write("DataFrame info:")
            st.write(df.info())
        return None

# Load the dataset
df = load_data()

if df is not None:
    try:
        # Get most recent data for current metrics
        current_data = df.iloc[-1]
        
        # Create dashboard layout
        # ----------------------
        
        # Current Metrics Section
        col1, col2, col3, col4 = st.columns(4)
        
        # Display current load with delta
        with col1:
            st.metric(
                label="Current Load",
                value=f"{current_data['load']:.2f} MW",
                delta=f"{current_data['load'] - df.iloc[-2]['load']:.2f} MW"
            )
        
        # Display renewable energy metrics
        with col2:
            renewable_energy = current_data['solar_generation'] + current_data['wind_generation']
            prev_renewable = df.iloc[-2]['solar_generation'] + df.iloc[-2]['wind_generation']
            st.metric(
                label="Renewable Energy",
                value=f"{renewable_energy:.2f} MW",
                delta=f"{renewable_energy - prev_renewable:.2f} MW",
                delta_color="normal"
            )
        
        # Display grid stability
        with col3:
            stability = current_data['grid_stability']
            prev_stability = df.iloc[-2]['grid_stability']
            st.metric(
                label="Grid Stability",
                value=f"{stability:.1f}%",
                delta=f"{stability - prev_stability:.1f}%",
                delta_color="inverse"
            )
        
        # Display current price
        with col4:
            price = current_data['price_per_kwh']
            prev_price = df.iloc[-2]['price_per_kwh']
            st.metric(
                label="Current Price",
                value=f"${price:.3f}/kWh",
                delta=f"${price - prev_price:.3f}",
                delta_color="off"
            )

        # Real-Time Energy Consumption Chart
        # ---------------------------------
        st.subheader("Real-Time Energy Consumption (Last 24 Hours)")
        
        # Get last 24 hours of data
        latest_time = df['timestamp'].max()
        last_24h = latest_time - pd.Timedelta(hours=24)
        realtime_data = df[df['timestamp'] >= last_24h].copy()
        
        # Create real-time consumption figure
        rt_fig = go.Figure()
        
        # Add total load line
        rt_fig.add_trace(go.Scatter(
            x=realtime_data['timestamp'],
            y=realtime_data['load'],
            name='Total Load',
            line=dict(color='#e74c3c', width=2)
        ))
        
        # Add renewable sources
        rt_fig.add_trace(go.Scatter(
            x=realtime_data['timestamp'],
            y=realtime_data['solar_generation'],
            name='Solar Generation',
            line=dict(color='#f1c40f', width=2)
        ))
        
        rt_fig.add_trace(go.Scatter(
            x=realtime_data['timestamp'],
            y=realtime_data['wind_generation'],
            name='Wind Generation',
            line=dict(color='#3498db', width=2)
        ))
        
        # Update layout for better visualization
        rt_fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Power (MW)',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            plot_bgcolor='white'
        )
        
        # Add grid lines
        rt_fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickformat='%H:%M\n%Y-%m-%d'
        )
        
        rt_fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        )
        
        st.plotly_chart(rt_fig, use_container_width=True)

        # Create two columns for additional charts
        chart_col1, chart_col2 = st.columns(2)

        # 7-Day Energy Consumption Chart
        # -----------------------------
        with chart_col1:
            st.subheader("7-Day Energy Consumption")
            
            # Calculate the timestamp for 7 days ago
            seven_days_ago = latest_time - pd.Timedelta(days=7)
            last_7d = df[df['timestamp'] >= seven_days_ago].copy()
            
            # Create daily aggregations
            daily_data = last_7d.groupby(last_7d['timestamp'].dt.date).agg({
                'load': ['mean', 'max', 'min'],
                'solar_generation': 'sum',
                'wind_generation': 'sum'
            }).reset_index()
            
            daily_data.columns = ['date', 'avg_load', 'max_load', 'min_load', 'solar_total', 'wind_total']
            
            # Create consumption figure
            fig = go.Figure()
            
            # Add load range area
            fig.add_trace(go.Scatter(
                x=daily_data['date'],
                y=daily_data['max_load'],
                name='Max Load',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_data['date'],
                y=daily_data['min_load'],
                name='Load Range',
                fill='tonexty',
                fillcolor='rgba(46, 204, 113, 0.2)',
                line=dict(width=0),
                showlegend=False
            ))
            
            # Add average load line
            fig.add_trace(go.Scatter(
                x=daily_data['date'],
                y=daily_data['avg_load'],
                name='Average Load',
                line=dict(color='#2ecc71', width=2)
            ))
            
            # Add renewable energy bars
            fig.add_trace(go.Bar(
                x=daily_data['date'],
                y=daily_data['solar_total'],
                name='Solar Generation',
                marker_color='#f1c40f',
                opacity=0.7
            ))
            
            fig.add_trace(go.Bar(
                x=daily_data['date'],
                y=daily_data['wind_total'],
                name='Wind Generation',
                marker_color='#3498db',
                opacity=0.7
            ))
            
            # Update layout
            fig.update_layout(
                title='Last 7 Days Energy Consumption and Generation',
                xaxis_title='Date',
                yaxis_title='Power (MW)',
                hovermode='x unified',
                barmode='stack',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                plot_bgcolor='white'
            )
            
            # Add grid lines
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickangle=45,
                dtick='D1'
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            )
            
            # Add statistics
            stats_cols = st.columns(4)
            
            # Calculate week-over-week changes
            prev_week_start = seven_days_ago - pd.Timedelta(days=7)
            prev_week_data = df[(df['timestamp'] >= prev_week_start) & (df['timestamp'] < seven_days_ago)]
            
            current_week_avg = daily_data['avg_load'].mean()
            prev_week_avg = prev_week_data['load'].mean()
            
            current_week_peak = daily_data['max_load'].max()
            prev_week_peak = prev_week_data['load'].max()
            
            current_renewable = (daily_data['solar_total'].sum() + daily_data['wind_total'].sum()) / 7
            prev_renewable = (prev_week_data['solar_generation'].sum() + prev_week_data['wind_generation'].sum()) / 7
            
            # Display weekly statistics
            with stats_cols[0]:
                st.metric(
                    "Average Daily Load",
                    f"{current_week_avg:.2f} MW",
                    f"{current_week_avg - prev_week_avg:.2f} MW vs Prev Week"
                )
            with stats_cols[1]:
                st.metric(
                    "Peak Load",
                    f"{current_week_peak:.2f} MW",
                    f"{current_week_peak - prev_week_peak:.2f} MW vs Prev Week"
                )
            with stats_cols[2]:
                st.metric(
                    "Load Factor",
                    f"{(current_week_avg / current_week_peak * 100):.1f}%",
                    f"{(current_week_avg / current_week_peak * 100) - (prev_week_avg / prev_week_peak * 100):.1f}%"
                )
            with stats_cols[3]:
                st.metric(
                    "Avg Daily Renewable",
                    f"{current_renewable:.2f} MW",
                    f"{current_renewable - prev_renewable:.2f} MW vs Prev Week"
                )
            
            st.plotly_chart(fig, use_container_width=True)

        # Energy Source Distribution Chart
        # -------------------------------
        with chart_col2:
            st.subheader("Energy Source Distribution")
            latest_data = df.iloc[-1]
            sources = ['Solar', 'Wind', 'Natural Gas', 'Coal', 'Nuclear']
            values = [
                latest_data['solar_generation'],
                latest_data['wind_generation'],
                latest_data['natural_gas_generation'],
                latest_data['coal_generation'],
                latest_data['nuclear_generation']
            ]
            
            # Create donut chart
            fig = go.Figure(data=[go.Pie(
                labels=sources,
                values=values,
                hole=0.4,
                marker=dict(
                    colors=['#f1c40f', '#3498db', '#e67e22', '#7f8c8d', '#2ecc71']
                )
            )])
            
            fig.update_layout(
                title='Current Energy Source Distribution',
                annotations=[dict(text='Total<br>' + f"{sum(values):.1f} MW", x=0.5, y=0.5, font_size=12, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add source percentages
            total_energy = sum(values)
            st.write("Energy Source Breakdown:")
            source_cols = st.columns(len(sources))
            for i, (source, value) in enumerate(zip(sources, values)):
                with source_cols[i]:
                    percentage = (value / total_energy) * 100
                    st.metric(
                        source,
                        f"{percentage:.1f}%",
                        f"{value:.1f} MW"
                    )

        # Model Prediction Section
        # -----------------------
        st.markdown("---")
        st.header("Load Prediction Models")
        
        # Create columns for model settings
        model_col1, model_col2 = st.columns([1, 2])
        
        with model_col1:
            # Model selection dropdown
            model_type = st.selectbox(
                "Select Model",
                ["Random Forest", "Linear Regression", "Support Vector Regression", "XGBoost"]
            )
            
            # Feature selection
            feature_options = [
                'temperature', 'humidity', 'solar_generation', 'wind_generation',
                'grid_stability', 'power_factor', 'time_of_day', 'day_of_week',
                'voltage', 'current', 'price_per_kwh'
            ]
            selected_features = st.multiselect(
                "Select Features for Prediction",
                feature_options,
                default=['temperature', 'humidity', 'solar_generation', 'wind_generation', 'voltage', 'current', 'price_per_kwh']
            )
            
            # Add time features
            df['time_of_day'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Training window selection
            training_days = st.slider(
                "Training Window (Days)",
                min_value=1,
                max_value=30,
                value=7,
                help="Number of days of historical data to use for training"
            )
            
            # Train model button
            train_model = st.button("Train Model")

        with model_col2:
            if train_model and selected_features:
                try:
                    # Prepare the data
                    cutoff_date = latest_time - pd.Timedelta(days=training_days)
                    train_data = df[df['timestamp'] >= cutoff_date].copy()
                    
                    # Store training data in session state
                    st.session_state.training_data = train_data
                    
                    # Prepare features and target
                    X = train_data[selected_features]
                    y = train_data['load']
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Scale the features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Store the scaler in session state
                    st.session_state.feature_scaler = scaler
                    
                    # Initialize and train the selected model
                    if model_type == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_type == "Linear Regression":
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                    elif model_type == "Support Vector Regression":
                        from sklearn.svm import SVR
                        model = SVR(kernel='rbf')
                    else:  # XGBoost
                        import xgboost as xgb
                        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                    
                    model.fit(X_train_scaled, y_train)
                    
                    # Store the model and features in session state
                    st.session_state.trained_model = model
                    st.session_state.selected_features = selected_features
                    st.session_state.model_type = model_type
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate all metrics for current model
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                    
                    # Add Model Performance Section
                    st.subheader("Model Performance Analysis")
                    
                    # Display current model metrics
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric(
                            "R² Score",
                            f"{r2:.3f}",
                            help="Coefficient of determination (1 is perfect prediction)"
                        )
                    
                    with metric_cols[1]:
                        st.metric(
                            "RMSE",
                            f"{rmse:.2f} MW",
                            help="Root Mean Square Error"
                        )
                    
                    with metric_cols[2]:
                        st.metric(
                            "MAE",
                            f"{mae:.2f} MW",
                            help="Mean Absolute Error"
                        )
                    
                    with metric_cols[3]:
                        st.metric(
                            "MAPE",
                            f"{mape:.2f}%",
                            help="Mean Absolute Percentage Error"
                        )
                    
                    # Add Prediction vs Actual Visualization
                    st.markdown("### Prediction vs Actual Load Comparison")
                    
                    # Create tabs for different visualizations
                    compare_tabs = st.tabs(["Time Series View", "Scatter Plot", "Error Analysis"])
                    
                    with compare_tabs[0]:
                        # Time series view of predictions vs actual values
                        fig = go.Figure()
                        
                        # Add actual values
                        fig.add_trace(go.Scatter(
                            x=list(range(len(y_test))),
                            y=y_test.values,
                            name='Actual Load',
                            line=dict(color='#2ecc71', width=2)
                        ))
                        
                        # Add predicted values
                        fig.add_trace(go.Scatter(
                            x=list(range(len(y_pred))),
                            y=y_pred,
                            name='Predicted Load',
                            line=dict(color='#e74c3c', width=2, dash='dash')
                        ))
                        
                        # Add error bands
                        error = np.abs(y_test.values - y_pred)
                        fig.add_trace(go.Scatter(
                            x=list(range(len(y_pred))),
                            y=y_pred + error,
                            fill=None,
                            mode='lines',
                            line_color='rgba(231, 76, 60, 0)',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(len(y_pred))),
                            y=y_pred - error,
                            fill='tonexty',
                            mode='lines',
                            name='Error Range',
                            line_color='rgba(231, 76, 60, 0)',
                            fillcolor='rgba(231, 76, 60, 0.2)'
                        ))
                        
                        fig.update_layout(
                            title='Time Series: Predicted vs Actual Load',
                            xaxis_title='Time Point',
                            yaxis_title='Load (MW)',
                            hovermode='x unified',
                            plot_bgcolor='white'
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add summary statistics
                        st.markdown("#### Time Series Statistics")
                        ts_cols = st.columns(4)
                        with ts_cols[0]:
                            st.metric("Mean Actual Load", f"{y_test.mean():.2f} MW")
                        with ts_cols[1]:
                            st.metric("Mean Predicted Load", f"{y_pred.mean():.2f} MW")
                        with ts_cols[2]:
                            st.metric("Mean Error", f"{error.mean():.2f} MW")
                        with ts_cols[3]:
                            st.metric("Error Std Dev", f"{error.std():.2f} MW")
                    
                    with compare_tabs[1]:
                        # Scatter plot of predicted vs actual values
                        fig = go.Figure()
                        
                        # Add perfect prediction line
                        min_val = min(min(y_test), min(y_pred))
                        max_val = max(max(y_test), max(y_pred))
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash'),
                            showlegend=True
                        ))
                        
                        # Add scatter plot
                        fig.add_trace(go.Scatter(
                            x=y_test,
                            y=y_pred,
                            mode='markers',
                            name='Predictions',
                            marker=dict(
                                color='#2ecc71',
                                size=8,
                                opacity=0.6
                            )
                        ))
                        
                        fig.update_layout(
                            title='Scatter Plot: Predicted vs Actual Load',
                            xaxis_title='Actual Load (MW)',
                            yaxis_title='Predicted Load (MW)',
                            plot_bgcolor='white'
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add regression statistics
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
                        
                        st.markdown("#### Regression Statistics")
                        reg_cols = st.columns(4)
                        with reg_cols[0]:
                            st.metric("Slope", f"{slope:.3f}")
                        with reg_cols[1]:
                            st.metric("Intercept", f"{intercept:.2f}")
                        with reg_cols[2]:
                            st.metric("R-value", f"{r_value:.3f}")
                        with reg_cols[3]:
                            st.metric("Standard Error", f"{std_err:.3f}")
                    
                    with compare_tabs[2]:
                        # Error analysis
                        residuals = y_test.values - y_pred
                        
                        # Create error distribution plot
                        fig = go.Figure()
                        
                        # Add histogram of errors
                        fig.add_trace(go.Histogram(
                            x=residuals,
                            name='Error Distribution',
                            nbinsx=30,
                            marker_color='#3498db'
                        ))
                        
                        # Add vertical line for mean error
                        fig.add_vline(
                            x=residuals.mean(),
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Mean Error: {residuals.mean():.2f}",
                            annotation_position="top right"
                        )
                        
                        fig.update_layout(
                            title='Distribution of Prediction Errors',
                            xaxis_title='Prediction Error (MW)',
                            yaxis_title='Frequency',
                            plot_bgcolor='white',
                            showlegend=False
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add error statistics
                        st.markdown("#### Error Statistics")
                        error_cols = st.columns(4)
                        with error_cols[0]:
                            st.metric("Mean Error", f"{residuals.mean():.2f} MW")
                        with error_cols[1]:
                            st.metric("Error Std Dev", f"{residuals.std():.2f} MW")
                        with error_cols[2]:
                            st.metric("Skewness", f"{stats.skew(residuals):.3f}")
                        with error_cols[3]:
                            st.metric("Kurtosis", f"{stats.kurtosis(residuals):.3f}")
                    
                    # Training Window Impact Analysis
                    st.markdown("---")
                    st.subheader("Training Window Impact Analysis")
                    
                    # Create different window sizes to test
                    window_sizes = [1, 3, 7, 14, 21, 30]  # days
                    window_metrics = []
                    
                    # Progress bar for window analysis
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Analyze each window size
                    for i, window in enumerate(window_sizes):
                        status_text.text(f"Analyzing {window}-day window...")
                        
                        # Get data for this window
                        window_cutoff = latest_time - pd.Timedelta(days=window)
                        window_data = df[df['timestamp'] >= window_cutoff].copy()
                        
                        # Prepare features and target
                        X_window = window_data[selected_features]
                        y_window = window_data['load']
                        
                        # Split and scale data
                        X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
                            X_window, y_window, test_size=0.2, random_state=42
                        )
                        
                        scaler_w = StandardScaler()
                        X_train_w_scaled = scaler_w.fit_transform(X_train_w)
                        X_test_w_scaled = scaler_w.transform(X_test_w)
                        
                        # Train model
                        if model_type == "Random Forest":
                            window_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif model_type == "Linear Regression":
                            window_model = LinearRegression()
                        elif model_type == "Support Vector Regression":
                            window_model = SVR(kernel='rbf')
                        else:  # XGBoost
                            window_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                        
                        window_model.fit(X_train_w_scaled, y_train_w)
                        
                        # Make predictions
                        y_pred_w = window_model.predict(X_test_w_scaled)
                        
                        # Calculate metrics
                        r2_w = r2_score(y_test_w, y_pred_w)
                        rmse_w = np.sqrt(mean_squared_error(y_test_w, y_pred_w))
                        mae_w = mean_absolute_error(y_test_w, y_pred_w)
                        mape_w = mean_absolute_percentage_error(y_test_w, y_pred_w) * 100
                        
                        window_metrics.append({
                            'Window Size': window,
                            'R² Score': r2_w,
                            'RMSE': rmse_w,
                            'MAE': mae_w,
                            'MAPE': mape_w,
                            'Training Samples': len(X_train_w),
                            'Testing Samples': len(X_test_w)
                        })
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(window_sizes))
                    
                    status_text.text("Analysis complete!")
                    
                    # Convert results to DataFrame
                    window_df = pd.DataFrame(window_metrics)
                    
                    # Create tabs for different visualizations
                    window_tabs = st.tabs(["Metrics Comparison", "Sample Size Impact", "Detailed Metrics"])
                    
                    with window_tabs[0]:
                        # Create multi-metric line plot
                        fig = go.Figure()
                        
                        # Add traces for each metric
                        metrics_to_plot = ['R² Score', 'RMSE', 'MAE', 'MAPE']
                        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f']
                        
                        for metric, color in zip(metrics_to_plot, colors):
                            # Normalize the metric for comparison
                            if metric != 'R² Score':  # These should be minimized
                                normalized = (window_df[metric] - window_df[metric].min()) / (window_df[metric].max() - window_df[metric].min())
                                normalized = 1 - normalized  # Invert so higher is better
                            else:  # R² should be maximized
                                normalized = (window_df[metric] - window_df[metric].min()) / (window_df[metric].max() - window_df[metric].min())
                            
                            fig.add_trace(go.Scatter(
                                x=window_df['Window Size'],
                                y=normalized,
                                name=metric,
                                line=dict(color=color, width=2),
                                hovertemplate=f"{metric}: %{{text}}<br>Window Size: %{{x}} days<extra></extra>",
                                text=[f"{val:.3f}" for val in window_df[metric]]
                            ))
                        
                        fig.update_layout(
                            title='Impact of Training Window Size on Model Performance',
                            xaxis_title='Window Size (Days)',
                            yaxis_title='Normalized Performance (Higher is Better)',
                            plot_bgcolor='white',
                            hovermode='x unified'
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', dtick=1)
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with window_tabs[1]:
                        # Create sample size vs performance plot
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=window_df['Training Samples'],
                            y=window_df['R² Score'],
                            name='R² Score',
                            mode='lines+markers',
                            line=dict(color='#2ecc71', width=2),
                            marker=dict(size=8)
                        ))
                        
                        # Add window size annotations
                        for i, row in window_df.iterrows():
                            fig.add_annotation(
                                x=row['Training Samples'],
                                y=row['R² Score'],
                                text=f"{row['Window Size']}d",
                                showarrow=True,
                                arrowhead=1,
                                ax=0,
                                ay=-30
                            )
                        
                        fig.update_layout(
                            title='Training Sample Size vs Model Performance',
                            xaxis_title='Number of Training Samples',
                            yaxis_title='R² Score',
                            plot_bgcolor='white'
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with window_tabs[2]:
                        # Display detailed metrics table
                        st.dataframe(
                            window_df.style.format({
                                'R² Score': '{:.3f}',
                                'RMSE': '{:.2f}',
                                'MAE': '{:.2f}',
                                'MAPE': '{:.2f}%'
                            }),
                            use_container_width=True
                        )
                        
                        # Find optimal window size
                        optimal_window = window_df.loc[window_df['R² Score'].idxmax()]
                        st.info(f"Optimal Training Window: {optimal_window['Window Size']} days "
                               f"(R² Score: {optimal_window['R² Score']:.3f}, "
                               f"RMSE: {optimal_window['RMSE']:.2f} MW)")
                    
                    # Add recommendations based on analysis
                    st.markdown("### Training Window Recommendations")
                    
                    # Calculate stability metrics
                    r2_std = window_df['R² Score'].std()
                    r2_range = window_df['R² Score'].max() - window_df['R² Score'].min()
                    
                    recommendations = []
                    
                    if r2_range < 0.1:
                        recommendations.append("Model performance is relatively stable across different window sizes.")
                    else:
                        recommendations.append("Model performance varies significantly with window size.")
                    
                    if window_df['R² Score'].iloc[-1] > window_df['R² Score'].iloc[0]:
                        recommendations.append("Longer training windows generally improve model performance.")
                    else:
                        recommendations.append("Shorter training windows might be sufficient for good performance.")
                    
                    if r2_std < 0.05:
                        recommendations.append("Model is robust to window size changes.")
                    else:
                        recommendations.append("Model is sensitive to window size selection.")
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                
                    # Add Feature Importance Analysis Section
                    st.markdown("---")
                    st.subheader("Feature Importance Analysis")
                    
                    # Get feature importance based on model type
                    if model_type == "Random Forest":
                        importance_scores = model.feature_importances_
                        importance_type = "Feature Importance"
                    elif model_type == "Linear Regression":
                        importance_scores = np.abs(model.coef_)
                        importance_type = "Absolute Coefficient"
                    elif model_type == "XGBoost":
                        importance_scores = model.feature_importances_
                        importance_type = "Feature Importance"
                    else:  # SVR - use permutation importance
                        from sklearn.inspection import permutation_importance
                        # Calculate permutation importance
                        perm_importance = permutation_importance(
                            model, 
                            X_test_scaled, 
                            y_test,
                            n_repeats=5,
                            random_state=42
                        )
                        importance_scores = perm_importance.importances_mean
                        importance_type = "Permutation Importance"
                    
                    # Create feature importance DataFrame
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': importance_scores
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=True)
                    
                    # Create tabs for different visualizations
                    importance_tabs = st.tabs(["Bar Chart", "Correlation Analysis", "Feature Statistics"])
                    
                    with importance_tabs[0]:
                        # Create horizontal bar chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=importance_df['Importance'],
                            y=importance_df['Feature'],
                            orientation='h',
                            marker=dict(
                                color=importance_df['Importance'],
                                colorscale='Viridis'
                            )
                        ))
                        
                        fig.update_layout(
                            title=f'Feature Importance ({model_type})',
                            xaxis_title=importance_type,
                            yaxis_title='Feature',
                            plot_bgcolor='white'
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add importance statistics
                        st.markdown("#### Top Features Analysis")
                        top_features = importance_df.iloc[::-1][:3]  # Get top 3 features
                        for _, row in top_features.iterrows():
                            st.markdown(f"- **{row['Feature']}**: {row['Importance']:.4f}")
                    
                    with importance_tabs[1]:
                        # Calculate correlation matrix
                        correlation_matrix = st.session_state.training_data[selected_features + ['load']].corr()
                        
                        # Create heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=correlation_matrix,
                            x=correlation_matrix.columns,
                            y=correlation_matrix.columns,
                            colorscale='RdBu',
                            zmin=-1,
                            zmax=1,
                            text=np.round(correlation_matrix, 2),
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            hoverongaps=False
                        ))
                        
                        fig.update_layout(
                            title='Feature Correlation Matrix',
                            plot_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add correlation insights
                        st.markdown("#### Correlation Insights")
                        load_correlations = correlation_matrix['load'].drop('load').sort_values(ascending=False)
                        
                        st.markdown("Strongest correlations with load:")
                        for feat, corr in load_correlations.items():
                            st.markdown(f"- **{feat}**: {corr:.3f}")
                    
                    with importance_tabs[2]:
                        # Calculate feature statistics
                        stats_df = st.session_state.training_data[selected_features].describe()
                        
                        # Add additional statistics
                        stats_df.loc['skew'] = st.session_state.training_data[selected_features].skew()
                        stats_df.loc['kurtosis'] = st.session_state.training_data[selected_features].kurtosis()
                        
                        # Create feature statistics table
                        st.dataframe(
                            stats_df.style.format("{:.2f}"),
                            use_container_width=True
                        )
                        
                        # Add distribution plots
                        st.markdown("#### Feature Distributions")
                        
                        # Create subplots for distributions
                        num_features = len(selected_features)
                        num_cols = 2
                        num_rows = (num_features + 1) // 2
                        
                        fig = go.Figure()
                        
                        for feature in selected_features:
                            # Add histogram for each feature
                            fig.add_trace(go.Histogram(
                                x=st.session_state.training_data[feature],
                                name=feature,
                                nbinsx=30,
                                opacity=0.7
                            ))
                        
                        fig.update_layout(
                            title='Feature Distributions',
                            barmode='overlay',
                            plot_bgcolor='white',
                            showlegend=True,
                            height=400
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    st.write("Error details:", e.__class__.__name__)

            # Prediction Section - Moved outside the training block
            if st.session_state.trained_model is not None:
                st.markdown("---")
                st.subheader("Make New Predictions")
                
                # Create columns for input features
                input_cols = st.columns(4)  # Show 4 inputs per row
                feature_inputs = {}
                
                # Create input fields for each selected feature
                for i, feature in enumerate(st.session_state.selected_features):
                    col_idx = i % 4
                    with input_cols[col_idx]:
                        if feature == 'time_of_day':
                            time_input = st.time_input(
                                "Time of Day",
                                value=datetime.now().time()
                            )
                            feature_inputs[feature] = time_input.hour + time_input.minute / 60
                        elif feature == 'day_of_week':
                            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            selected_day = st.selectbox(
                                "Day of Week",
                                day_names,
                                index=datetime.now().weekday()
                            )
                            feature_inputs[feature] = day_names.index(selected_day)
                        else:
                            feat_mean = st.session_state.training_data[feature].mean()
                            feat_std = st.session_state.training_data[feature].std()
                            feat_min = st.session_state.training_data[feature].min()
                            feat_max = st.session_state.training_data[feature].max()
                            
                            feature_inputs[feature] = st.number_input(
                                f"{feature.replace('_', ' ').title()}",
                                min_value=float(feat_min),
                                max_value=float(feat_max),
                                value=float(feat_mean),
                                help=f"Range: [{feat_min:.2f}, {feat_max:.2f}], Mean: {feat_mean:.2f}, Std: {feat_std:.2f}",
                                key=f"input_{feature}"  # Add unique key for each input
                            )

                # Add a predict button
                if st.button("Predict Load", key="predict_button"):
                    try:
                        # Prepare input data
                        input_data = pd.DataFrame([feature_inputs])
                        
                        # Scale the input data using the saved scaler
                        input_scaled = st.session_state.feature_scaler.transform(input_data)
                        
                        # Make prediction using saved model
                        prediction = st.session_state.trained_model.predict(input_scaled)[0]
                        
                        # Display prediction with confidence interval for Random Forest
                        if st.session_state.model_type == "Random Forest":
                            predictions = []
                            for estimator in st.session_state.trained_model.estimators_:
                                predictions.append(estimator.predict(input_scaled)[0])
                            
                            pred_std = np.std(predictions)
                            conf_interval = 1.96 * pred_std  # 95% confidence interval
                            
                            # Create three columns for prediction display
                            pred_col1, pred_col2, pred_col3 = st.columns(3)
                            
                            with pred_col1:
                                st.metric(
                                    "Predicted Load",
                                    f"{prediction:.2f} MW"
                                )
                            with pred_col2:
                                st.metric(
                                    "Lower Bound (95% CI)",
                                    f"{(prediction - conf_interval):.2f} MW"
                                )
                            with pred_col3:
                                st.metric(
                                    "Upper Bound (95% CI)",
                                    f"{(prediction + conf_interval):.2f} MW"
                                )
                        else:
                            # For other models, just show the prediction
                            st.metric(
                                "Predicted Load",
                                f"{prediction:.2f} MW"
                            )
                        
                        # Add a gauge chart for the prediction
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Predicted Load (MW)"},
                            gauge={
                                'axis': {'range': [None, st.session_state.training_data['load'].max()]},
                                'bar': {'color': "#2ecc71"},
                                'steps': [
                                    {'range': [0, st.session_state.training_data['load'].mean()], 'color': "lightgray"},
                                    {'range': [st.session_state.training_data['load'].mean(), st.session_state.training_data['load'].max()], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': st.session_state.training_data['load'].mean()
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.write("Error details:", e.__class__.__name__)
            else:
                st.info("Train a model first to make predictions.")

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.write("Error details:", e.__class__.__name__)
else:
    st.warning("No data available. Please check if the dataset file exists and is accessible.") 