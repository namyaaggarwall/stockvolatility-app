"""
Full Dashboard: Stock Volatility Forecasting — GARCH vs LSTM vs Classification
Run locally with: streamlit run app.py

Required files in the same folder as this script:
- reliance_cleaned.csv
- garch_test_results.csv
- lstm_test_results_v2.csv
- classification_test_results.csv
- volatility_thresholds.pkl
- feature_importance.png
- classification_confusion_matrices.png
- lstm_volatility_model.keras
- scaler_returns.pkl, scaler_vol.pkl

(yfinance, arch, tensorflow must be pip installed locally for the
Future Forecast tab's live data features to work)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                               precision_score, recall_score, f1_score)

st.set_page_config(page_title="Stock Volatility Forecasting Dashboard", layout="wide")

st.title("Stock Volatility Forecasting Dashboard")
st.caption("RELIANCE stock | GARCH vs LSTM vs Classification")

tab1, tab2, tab3 = st.tabs(["Historical Comparison", "Classification Model", "Future Forecast"])

# ==============================================================
# TAB 1 — Historical Comparison (GARCH vs LSTM, 2019-2021)
# ==============================================================
with tab1:
    @st.cache_data
    def load_historical():
        garch = pd.read_csv('garch_test_results.csv', parse_dates=['Date'])
        lstm = pd.read_csv('lstm_test_results_v2.csv', parse_dates=['Date'])
        merged = garch[['Date', 'Volatility_20d', 'GARCH_Predicted_Volatility']].merge(
            lstm[['Date', 'LSTM_Predicted_Volatility']], on='Date'
        )
        merged.columns = ['Date', 'Actual_Volatility', 'GARCH_Predicted', 'LSTM_Predicted']
        return merged

    data = load_historical()

    st.sidebar.header("Historical View Controls")
    min_date, max_date = data['Date'].min(), data['Date'].max()

    if "date_range_widget" not in st.session_state:
        st.session_state["date_range_widget"] = (min_date, max_date)

    st.sidebar.markdown("**Quick jump to known events:**")
    if st.sidebar.button("COVID Crash (Mar 2020)"):
        st.session_state["date_range_widget"] = (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-07-01'))

    date_range = st.sidebar.date_input(
        "Select date range", min_value=min_date, max_value=max_date, key="date_range_widget"
    )
    show_garch = st.sidebar.checkbox("Show GARCH prediction", value=True)
    show_lstm = st.sidebar.checkbox("Show LSTM prediction", value=True)

    if len(date_range) == 2:
        start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    else:
        filtered = data

    st.subheader("Actual vs Predicted Volatility (Test Period)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(filtered['Date'], filtered['Actual_Volatility'], label='Actual Volatility', color='black', linewidth=2)
    if show_garch:
        ax.plot(filtered['Date'], filtered['GARCH_Predicted'], label='GARCH Predicted', color='orange', alpha=0.8)
    if show_lstm:
        ax.plot(filtered['Date'], filtered['LSTM_Predicted'], label='LSTM Predicted', color='teal', alpha=0.8)
    ax.set_xlabel("Date"); ax.set_ylabel("Volatility"); ax.legend()
    st.pyplot(fig)

    st.subheader("Model Performance (selected period)")
    col1, col2 = st.columns(2)
    if len(filtered) > 0:
        with col1:
            st.markdown("**GARCH(1,1)**")
            mae_g = mean_absolute_error(filtered['Actual_Volatility'], filtered['GARCH_Predicted'])
            rmse_g = np.sqrt(mean_squared_error(filtered['Actual_Volatility'], filtered['GARCH_Predicted']))
            r2_g = r2_score(filtered['Actual_Volatility'], filtered['GARCH_Predicted'])
            st.metric("MAE", f"{mae_g:.5f}"); st.metric("RMSE", f"{rmse_g:.5f}"); st.metric("R²", f"{r2_g:.4f}")
        with col2:
            st.markdown("**LSTM**")
            mae_l = mean_absolute_error(filtered['Actual_Volatility'], filtered['LSTM_Predicted'])
            rmse_l = np.sqrt(mean_squared_error(filtered['Actual_Volatility'], filtered['LSTM_Predicted']))
            r2_l = r2_score(filtered['Actual_Volatility'], filtered['LSTM_Predicted'])
            st.metric("MAE", f"{mae_l:.5f}"); st.metric("RMSE", f"{rmse_l:.5f}"); st.metric("R²", f"{r2_l:.4f}")

        if mae_l < mae_g:
            st.success(f"LSTM outperforms GARCH here (lower MAE by {mae_g - mae_l:.5f})")
        else:
            st.info(f"GARCH outperforms LSTM here (lower MAE by {mae_l - mae_g:.5f})")

    with st.expander("View raw data"):
        st.dataframe(filtered)

# ==============================================================
# TAB 2 — Classification Model (Low / Medium / High)
# ==============================================================
with tab2:
    st.subheader("Volatility Category Classification")
    st.markdown("""
    Instead of an exact number, this model predicts whether a day's volatility
    falls into **Low**, **Medium**, or **High** — often more directly useful
    for quick risk-assessment decisions.
    """)

    @st.cache_data
    def load_classification():
        return pd.read_csv('classification_test_results.csv', parse_dates=['Date'])

    clf_results = load_classification()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Logistic Regression**")
        acc_lr = accuracy_score(clf_results['Actual_Category'], clf_results['LogReg_Predicted'])
        prec_lr = precision_score(clf_results['Actual_Category'], clf_results['LogReg_Predicted'], average='macro')
        rec_lr = recall_score(clf_results['Actual_Category'], clf_results['LogReg_Predicted'], average='macro')
        f1_lr = f1_score(clf_results['Actual_Category'], clf_results['LogReg_Predicted'], average='macro')
        st.metric("Accuracy", f"{acc_lr:.3f}")
        st.metric("Precision (macro)", f"{prec_lr:.3f}")
        st.metric("Recall (macro)", f"{rec_lr:.3f}")
        st.metric("F1-score (macro)", f"{f1_lr:.3f}")

    with col2:
        st.markdown("**Random Forest** (used for live predictions)")
        acc_rf = accuracy_score(clf_results['Actual_Category'], clf_results['RF_Predicted'])
        prec_rf = precision_score(clf_results['Actual_Category'], clf_results['RF_Predicted'], average='macro')
        rec_rf = recall_score(clf_results['Actual_Category'], clf_results['RF_Predicted'], average='macro')
        f1_rf = f1_score(clf_results['Actual_Category'], clf_results['RF_Predicted'], average='macro')
        st.metric("Accuracy", f"{acc_rf:.3f}")
        st.metric("Precision (macro)", f"{prec_rf:.3f}")
        st.metric("Recall (macro)", f"{rec_rf:.3f}")
        st.metric("F1-score (macro)", f"{f1_rf:.3f}")

    st.success(f"Random Forest selected as the primary classifier — {acc_rf:.3f} vs {acc_lr:.3f} accuracy.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Confusion Matrices**")
        try:
            st.image('classification_confusion_matrices.png')
        except Exception:
            st.info("classification_confusion_matrices.png not found in folder.")
    with col4:
        st.markdown("**What drives the prediction?**")
        try:
            st.image('feature_importance.png')
        except Exception:
            st.info("feature_importance.png not found in folder.")
        st.caption("Yesterday's volatility is by far the strongest driver — consistent with GARCH's own finding that volatility persists over time.")

    with st.expander("View raw classification results"):
        st.dataframe(clf_results)

# ==============================================================
# TAB 3 — Future Forecast (flexible N-day-ahead, live data)
# ==============================================================
with tab3:
    st.subheader("Forecast Future Volatility")
    st.markdown("""
    Pulls **live, current RELIANCE data** and forecasts volatility for as many
    days ahead as you choose — with a Low/Medium/High classification and a
    plain-English reason for each day.
    """)

    n_days = st.slider("How many days ahead do you want to forecast?", min_value=1, max_value=30, value=10)

    if n_days > 10:
        st.warning(
            "Forecasts beyond ~10 days should be treated with caution. GARCH's multi-day "
            "forecasts drift toward the long-run historical average (pulled up by past crashes "
            "like 2008 and COVID), while our LSTM's recursive forecast assumes recent conditions "
            "continue. The two models can diverge meaningfully at longer horizons — this is expected "
            "model behavior, not an error, but shorter horizons are more reliable."
        )

    if st.button("Fetch Live Data & Forecast"):
        with st.spinner("Pulling live data and generating forecast..."):
            try:
                import yfinance as yf
                from arch import arch_model
                from tensorflow.keras.models import load_model

                # 1. Pull live data
                live_df = yf.download("RELIANCE.NS", period="150d", interval="1d")
                live_df = live_df[['Close']].reset_index()
                live_df.columns = ['Date', 'Close']
                live_df['Return'] = live_df['Close'].pct_change()
                live_df['Volatility_20d'] = live_df['Return'].rolling(window=20).std()
                live_df = live_df.dropna().reset_index(drop=True)

                last_date = live_df['Date'].iloc[-1]
                st.write(f"Most recent live data: **{last_date.date()}**")

                # 2. Load classifier thresholds
                with open('volatility_thresholds.pkl', 'rb') as f:
                    thresholds = pickle.load(f)
                low_thresh, high_thresh = thresholds['low'], thresholds['high']

                def categorize(vol):
                    if vol < low_thresh:
                        return 'Low'
                    elif vol < high_thresh:
                        return 'Medium'
                    return 'High'

                # 3. GARCH multi-day forecast
                returns_pct = live_df['Return'] * 100
                garch_fitted = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal').fit(disp='off')
                garch_forecast = garch_fitted.forecast(horizon=n_days)
                garch_preds = np.sqrt(garch_forecast.variance.values[-1, :]) / 100

                # 4. LSTM recursive forecast
                lstm_model = load_model('lstm_volatility_model.keras')
                with open('scaler_returns.pkl', 'rb') as f:
                    scaler_returns = pickle.load(f)
                with open('scaler_vol.pkl', 'rb') as f:
                    scaler_vol = pickle.load(f)

                returns_window = list(live_df['Return'].values[-30:])
                vol_window = list(live_df['Volatility_20d'].values[-30:])
                avg_recent_return = np.mean(returns_window)

                lstm_preds = []
                for _ in range(n_days):
                    r_scaled = scaler_returns.transform(np.array(returns_window[-30:]).reshape(-1, 1))
                    v_scaled = scaler_vol.transform(np.array(vol_window[-30:]).reshape(-1, 1))
                    X_input = np.hstack([r_scaled, v_scaled]).reshape(1, 30, 2)
                    pred_scaled = lstm_model.predict(X_input, verbose=0)
                    pred_vol = scaler_vol.inverse_transform(pred_scaled)[0, 0]
                    lstm_preds.append(pred_vol)
                    returns_window.append(avg_recent_return)
                    vol_window.append(pred_vol)
                lstm_preds = np.array(lstm_preds)

                # 5. Classify + reasoning
                current_vol = live_df['Volatility_20d'].values[-1]
                avg_return_5d = np.mean(live_df['Return'].values[-5:])

                def generate_reason(pred_vol, category, current_vol, avg_return_5d):
                    trend = "rising" if pred_vol > current_vol else "falling" if pred_vol < current_vol else "stable"
                    base = f"Predicted {category.upper()}, {trend} from the current level ({current_vol:.4f})."
                    if avg_return_5d < -0.01:
                        extra = " Recent returns have been notably negative, which historically precedes higher volatility."
                    elif avg_return_5d > 0.01:
                        extra = " Recent returns have been strongly positive, often associated with calmer conditions."
                    else:
                        extra = " Recent returns have been fairly stable, supporting this forecast."
                    return base + extra

                # Simple recommendation — one short line per category, easy to explain
                def generate_recommendation(category):
                    if category == 'Low':
                        return "Recommendation: Lower risk — a relatively safer time to hold or invest."
                    elif category == 'Medium':
                        return "Recommendation: Moderate risk — proceed with normal caution."
                    else:
                        return "Recommendation: Higher risk — consider caution or reducing exposure."

                lstm_categories = [categorize(v) for v in lstm_preds]
                garch_categories = [categorize(v) for v in garch_preds]
                reasons = [generate_reason(v, c, current_vol, avg_return_5d) for v, c in zip(lstm_preds, lstm_categories)]
                recommendations = [generate_recommendation(c) for c in lstm_categories]

                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'GARCH_Volatility': garch_preds,
                    'GARCH_Category': garch_categories,
                    'LSTM_Volatility': lstm_preds,
                    'LSTM_Category': lstm_categories,
                    'Reasoning': reasons,
                    'Recommendation': recommendations
                })

                # 6. Plot
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(forecast_df['Date'], forecast_df['GARCH_Volatility'], label='GARCH Forecast', color='orange', marker='o')
                ax.plot(forecast_df['Date'], forecast_df['LSTM_Volatility'], label='LSTM Forecast', color='teal', marker='o')
                ax.axhline(low_thresh, color='green', linestyle='--', alpha=0.4, label='Low/Medium threshold')
                ax.axhline(high_thresh, color='red', linestyle='--', alpha=0.4, label='Medium/High threshold')
                ax.set_title(f"{n_days}-Day Volatility Forecast")
                ax.set_xlabel("Date"); ax.set_ylabel("Predicted Volatility")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)

                # 7. Tomorrow's headline prediction
                st.markdown("### Tomorrow's Outlook")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("GARCH", f"{garch_preds[0]:.5f}", forecast_df['GARCH_Category'].iloc[0])
                with col2:
                    st.metric("LSTM", f"{lstm_preds[0]:.5f}", forecast_df['LSTM_Category'].iloc[0])
                st.info(reasons[0])
                st.warning(recommendations[0])

                # 8. Full table
                st.markdown(f"### Full {n_days}-Day Forecast")
                st.dataframe(forecast_df, use_container_width=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info(
                    "Make sure lstm_volatility_model.keras, scaler_returns.pkl, scaler_vol.pkl, "
                    "and volatility_thresholds.pkl are all in the same folder as this app, and that "
                    "yfinance, arch, and tensorflow are installed."
                )

st.markdown("---")
st.caption("Predictive Modelling | RELIANCE stock")
