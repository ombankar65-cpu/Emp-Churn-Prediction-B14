from flask import Flask, render_template_string, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('naive.pkl', 'rb'))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-light: #64748b;
            --border: #cbd5e1;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: var(--bg); color: var(--text); display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
        .container { background: var(--card-bg); width: 100%; max-width: 600px; padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05); }
        h1 { font-size: 24px; font-weight: 600; margin-bottom: 8px; color: var(--text); text-align: center; }
        p.subtitle { text-align: center; color: var(--text-light); font-size: 14px; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .form-group { margin-bottom: 20px; }
        .form-group.full { grid-column: span 2; }
        label { display: block; font-size: 13px; font-weight: 500; margin-bottom: 8px; color: var(--text); }
        input, select { width: 100%; padding: 10px 14px; border: 1px solid var(--border); border-radius: 8px; font-size: 14px; outline: none; transition: border-color 0.2s; background: #fff; }
        input:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1); }
        button { width: 100%; background: var(--primary); color: white; border: none; padding: 12px; font-size: 15px; font-weight: 500; border-radius: 8px; cursor: pointer; transition: background 0.2s; margin-top: 10px; }
        button:hover { background: var(--primary-hover); }
        .result-card { margin-top: 25px; padding: 16px; border-radius: 8px; background: #e0e7ff; color: #312e81; text-align: center; font-weight: 500; font-size: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Employee Status Predictor</h1>
        <p class="subtitle">Enter the details below to run the classification model</p>
        
        <form method="POST">
            <div class="grid">
                <div class="form-group">
                    <label for="Education">Education</label>
                    <select name="Education" required>
                        <option value="0">Bachelors</option>
                        <option value="1">Masters</option>
                        <option value="2">PHD</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="JoiningYear">Joining Year</label>
                    <input type="number" name="JoiningYear" placeholder="e.g. 2018" required>
                </div>

                <div class="form-group">
                    <label for="City">City</label>
                    <select name="City" required>
                        <option value="0">Bangalore</option>
                        <option value="1">Pune</option>
                        <option value="2">New Delhi</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="PaymentTier">Payment Tier</label>
                    <select name="PaymentTier" required>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="Age">Age</label>
                    <input type="number" name="Age" placeholder="e.g. 28" required>
                </div>

                <div class="form-group">
                    <label for="Gender">Gender</label>
                    <select name="Gender" required>
                        <option value="0">Male</option>
                        <option value="1">Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="EverBenched">Ever Benched</label>
                    <select name="EverBenched" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="ExperienceInCurrentDomain">Experience in Domain (Years)</label>
                    <input type="number" name="ExperienceInCurrentDomain" placeholder="e.g. 3" required>
                </div>
            </div>

            <button type="submit">Predict Status</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            Prediction Result: <strong>{{ prediction }}</strong>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features matching the model's requirements[cite: 1]
            features = [
                float(request.form['Education']),
                float(request.form['JoiningYear']),
                float(request.form['City']),
                float(request.form['PaymentTier']),
                float(request.form['Age']),
                float(request.form['Gender']),
                float(request.form['EverBenched']),
                float(request.form['ExperienceInCurrentDomain'])
            ]
            
            final_features = np.array([features])
            prediction_val = model.predict(final_features)[0]
            prediction = str(prediction_val)
        except Exception as e:
            prediction = f"Error in processing input: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
