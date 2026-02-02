from flask import Flask, render_template, request, send_file
import os

app = Flask(__name__)

# Model will be loaded when needed
model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Import ML libraries only when needed
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from keras.models import load_model
            import datetime as dt
            import yfinance as yf
            from sklearn.preprocessing import MinMaxScaler
            
            plt.style.use("fivethirtyeight")
            
            # Load model if not already loaded
            global model
            if model is None:
                model = load_model('stock_dl_model.h5')
            
            stock = request.form.get('stock', 'POWERGRID.NS')  # default stock
            
            # Fetch historical stock data
            start = dt.datetime(2000, 1, 1)
            end = dt.datetime(2024, 10, 1)
            df = yf.download(stock, start=start, end=end)
            
            # Descriptive statistics
            data_desc = df.describe()
            
            # Calculate EMAs
            ema20 = df['Close'].ewm(span=20, adjust=False).mean()
            ema50 = df['Close'].ewm(span=50, adjust=False).mean()
            ema100 = df['Close'].ewm(span=100, adjust=False).mean()
            ema200 = df['Close'].ewm(span=200, adjust=False).mean()
            
            # Split data and scale
            data_training = pd.DataFrame(df['Close'][:int(len(df)*0.7)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):])
            scaler = MinMaxScaler(feature_range=(0,1))
            data_training_array = scaler.fit_transform(data_training)
            
            # Prepare testing data
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)
            
            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])
            x_test, y_test = np.array(x_test), np.array(y_test)
            
            # Predict using LSTM
            y_predicted = model.predict(x_test)
            
            # Inverse scaling
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor
            
            # Plot EMAs and predictions
            def save_plot(x, y_list, labels, title, filename):
                fig, ax = plt.subplots(figsize=(12,6))
                for y, label in zip(y_list, labels):
                    ax.plot(y, label=label)
                ax.set_title(title)
                ax.set_xlabel("Time")
                ax.set_ylabel("Price")
                ax.legend()
                path = f"static/{filename}"
                fig.savefig(path)
                plt.close(fig)
                return path
            
            ema_chart_path = save_plot(df.index, [df['Close'], ema20, ema50], 
                                       ['Closing Price','EMA 20','EMA 50'],
                                       "Closing Price vs Time (20 & 50 Days EMA)",
                                       "ema_20_50.png")
            
            ema_chart_path_100_200 = save_plot(df.index, [df['Close'], ema100, ema200],
                                               ['Closing Price','EMA 100','EMA 200'],
                                               "Closing Price vs Time (100 & 200 Days EMA)",
                                               "ema_100_200.png")
            
            prediction_chart_path = save_plot(range(len(y_test)), [y_test, y_predicted],
                                              ['Original Price','Predicted Price'],
                                              "Prediction vs Original Trend",
                                              "stock_prediction.png")
            
            # Save CSV
            csv_file_path = f"static/{stock}_dataset.csv"
            df.to_csv(csv_file_path)
            
            # Render template
            return render_template('index.html',
                                   plot_path_ema_20_50=ema_chart_path,
                                   plot_path_ema_100_200=ema_chart_path_100_200,
                                   plot_path_prediction=prediction_chart_path,
                                   data_desc=data_desc.to_html(classes='table table-bordered'),
                                   dataset_link=csv_file_path)
        
        except ImportError as e:
            return render_template('index.html', error=f"ML libraries not available: {e}. Valentine's features still work!")
        except Exception as e:
            return render_template('index.html', error=f"Stock prediction error: {e}")
    
    return render_template('valentine.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

@app.route('/valentine')
def valentine():
    return render_template('valentine.html')

@app.route('/yes')
def yes():
    return render_template('yes.html')

if __name__ == '__main__':
    app.run(debug=True)
