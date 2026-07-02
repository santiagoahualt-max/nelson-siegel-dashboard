import os
import io
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from scipy.optimize import minimize, minimize_scalar

app = Flask(__name__, static_folder='.', static_url_path='')

# ---------------------------------------------------------
# Nelson-Siegel Model Formulation
# ---------------------------------------------------------

def nelson_siegel(tau, beta0, beta1, beta2, lmbda):
    tau = np.array(tau, dtype=float)
    if lmbda <= 0:
        return np.zeros_like(tau) + 1e10
        
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = (1.0 - np.exp(-tau / lmbda)) / (tau / lmbda)
        factor = np.where(tau == 0, 1.0, factor)
        term1 = factor
        term2 = factor - np.exp(-tau / lmbda)
        
    return beta0 + beta1 * term1 + beta2 * term2

def fit_nelson_siegel(tau, y):
    tau = np.array(tau, dtype=float)
    y = np.array(y, dtype=float)
    
    # 1. Grid search / scalar optimization over lambda
    def get_beta_and_rss(lmbda_val):
        if lmbda_val <= 0.01:
            return None, 1e10
            
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = (1.0 - np.exp(-tau / lmbda_val)) / (tau / lmbda_val)
            factor = np.where(tau == 0, 1.0, factor)
            term1 = factor
            term2 = factor - np.exp(-tau / lmbda_val)
            
        X = np.column_stack([np.ones_like(tau), term1, term2])
        try:
            beta, rss, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if len(rss) > 0:
                return beta, rss[0]
            else:
                y_pred = X @ beta
                return beta, np.sum((y - y_pred) ** 2)
        except np.linalg.LinAlgError:
            return None, 1e10

    def rss_objective(lmbda_val):
        _, rss = get_beta_and_rss(lmbda_val)
        return rss

    res_scalar = minimize_scalar(rss_objective, bounds=(0.1, 15.0), method='bounded')
    best_lmbda = res_scalar.x
    best_beta, _ = get_beta_and_rss(best_lmbda)
    
    if best_beta is None:
        best_beta = [np.mean(y), 0.0, 0.0]
        best_lmbda = 2.0

    # 2. SLSQP constrained optimization (beta0 + beta1 >= 0) to prevent negative short rate
    bounds = [
        (0.0, 50.0),       # beta0
        (-50.0, 50.0),     # beta1
        (-50.0, 50.0),     # beta2
        (0.1, 20.0)        # lambda
    ]
    constraints = [{'type': 'ineq', 'fun': lambda params: params[0] + params[1]}]
    initial_guess = [best_beta[0], best_beta[1], best_beta[2], best_lmbda]
    
    if initial_guess[0] + initial_guess[1] < 0:
        initial_guess[1] = -initial_guess[0] + 0.01
        
    def loss_function(params):
        b0, b1, b2, lmb = params
        y_pred = nelson_siegel(tau, b0, b1, b2, lmb)
        reg = 1e-6 * (b0**2 + b1**2 + b2**2)
        return np.sum((y - y_pred) ** 2) + reg

    res_4d = minimize(loss_function, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP', options={'maxiter': 1000})
    
    if res_4d.success:
        return res_4d.x
    else:
        return initial_guess

# ---------------------------------------------------------
# Numeric Cleaning
# ---------------------------------------------------------

def clean_value(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
        
    val_str = str(val).strip()
    is_percent = False
    
    if val_str.endswith('%'):
        is_percent = True
        val_str = val_str[:-1].strip()
        
    val_str = val_str.replace('$', '').replace(' ', '')
    
    if ',' in val_str:
        if '.' in val_str:
            val_str = val_str.replace('.', '')
        val_str = val_str.replace(',', '.')
        
    try:
        return float(val_str)
    except ValueError:
        return np.nan

# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    req_data = request.get_json()
    if not req_data or 'data' not in req_data:
        return jsonify({'success': False, 'error': 'No data received.'}), 400
        
    raw_text = req_data['data']
    if not raw_text.strip():
        return jsonify({'success': False, 'error': 'El input de datos está vacío.'}), 400
        
    # Optional caching: write to input_data.txt so user can see it on disk
    try:
        with open('input_data.txt', 'w', encoding='utf-8') as f:
            f.write(raw_text)
    except Exception as e:
        print(f"[!] Warning: Could not write input_data.txt: {e}")
        
    # Read string into Pandas
    try:
        # Try separators: tab, comma, semicolon
        df = None
        for sep in ['\t', ',', ';']:
            try:
                temp_df = pd.read_csv(io.StringIO(raw_text), sep=sep)
                if temp_df.shape[1] > 1:
                    df = temp_df
                    break
            except Exception:
                continue
                
        if df is None:
            return jsonify({'success': False, 'error': 'No se pudo estructurar la tabla. Asegúrate de copiar las columnas completas desde Excel.'}), 400
            
        # Clean column headers
        df.columns = df.columns.str.strip()
        
        # Find columns
        col_bono = None
        col_duration = None
        col_tir = None
        
        # 1. Ticker / Bono
        bono_patterns = ['bono', 'ticker', 'asset', 'activo', 'nombre', 'empresa']
        for p in bono_patterns:
            for col in df.columns:
                if p in col.lower():
                    col_bono = col
                    break
            if col_bono:
                break
                
        # 2. Modified Duration (tau)
        dur_patterns = ['m duration', 'mduration', 'modified duration', 'duracion mod', 'duration', 'duracion', 'tau']
        for p in dur_patterns:
            for col in df.columns:
                if p in col.lower():
                    col_duration = col
                    break
            if col_duration:
                break
                
        # 3. Yield / TIR (y)
        tir_patterns = ['% tir', 'tir', 'yield', 'tea', 'ytm', 'y']
        for p in tir_patterns:
            for col in df.columns:
                if p in col.lower():
                    col_tir = col
                    break
            if col_tir:
                break
                
        # Fallback to defaults
        if not col_bono:
            col_bono = df.columns[1] if df.shape[1] > 1 else df.columns[0]
        if not col_duration:
            col_duration = df.columns[7] if df.shape[1] > 7 else df.columns[0]
        if not col_tir:
            col_tir = df.columns[5] if df.shape[1] > 5 else df.columns[0]
            
        # Create working df
        working_df = pd.DataFrame()
        working_df['Bono'] = df[col_bono].astype(str).str.strip()
        working_df['M_Duration'] = df[col_duration].apply(clean_value)
        working_df['TIR'] = df[col_tir].apply(clean_value)
        
        # Add metadata columns if they exist
        for col in df.columns:
            if col not in [col_bono, col_duration, col_tir]:
                working_df[col] = df[col]
                
        # Drop NaN
        working_df = working_df.dropna(subset=['M_Duration', 'TIR']).reset_index(drop=True)
        
        if len(working_df) < 4:
            return jsonify({'success': False, 'error': f'Se necesitan al menos 4 activos con datos numéricos válidos. Solo se encontraron {len(working_df)}.'}), 400
            
        # Scale correction for Yields (if in decimals)
        mean_yield = working_df['TIR'].mean()
        if mean_yield < 0.20:
            working_df['TIR'] = working_df['TIR'] * 100.0
            
        tau = working_df['M_Duration'].values
        y = working_df['TIR'].values
        
        # Fit
        beta0, beta1, beta2, lmbda = fit_nelson_siegel(tau, y)
        
        # Predictions & Residuals
        working_df['Fitted_TIR'] = nelson_siegel(tau, beta0, beta1, beta2, lmbda)
        working_df['Residual'] = working_df['TIR'] - working_df['Fitted_TIR']
        working_df['Residual_bps'] = working_df['Residual'] * 100.0
        
        mean_res = working_df['Residual'].mean()
        std_res = working_df['Residual'].std(ddof=1) if len(working_df) > 1 else 1.0
        if std_res < 1e-6:
            std_res = 1.0
            
        working_df['Z_Score'] = (working_df['Residual'] - mean_res) / std_res
        
        def get_signal(z):
            if z > 1.5: return "Cheap (BUY)"
            elif z > 0.5: return "Slightly Cheap"
            elif z < -1.5: return "Rich (SELL)"
            elif z < -0.5: return "Slightly Rich"
            else: return "Neutral"
            
        working_df['Signal'] = working_df['Z_Score'].apply(get_signal)
        
        # Sort by Z-score (descending)
        working_df_sorted = working_df.sort_values(by='Z_Score', ascending=False)
        
        # Save output CSV
        output_file = "output_data.csv"
        try:
            working_df_sorted.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
        except PermissionError:
            pass  # Fail silently on auto-save if file is open in Excel
            
        # Compute RMSE
        rmse = float(np.sqrt(np.mean((y - working_df['Fitted_TIR'].values) ** 2)))
        
        # Prepare JSON response
        bonds_list = []
        for idx, row in working_df_sorted.iterrows():
            bond_data = {
                'ticker': str(row['Bono']),
                'duration': float(row['M_Duration']),
                'yield': float(row['TIR']),
                'fitted': float(row['Fitted_TIR']),
                'residual_bps': float(row['Residual_bps']),
                'z_score': float(row['Z_Score']),
                'signal': str(row['Signal'])
            }
            # Add extra columns for details
            extra_details = {}
            for col in working_df_sorted.columns:
                if col not in ['Bono', 'M_Duration', 'TIR', 'Fitted_TIR', 'Residual', 'Residual_bps', 'Z_Score', 'Signal']:
                    extra_details[col] = str(row[col])
            bond_data['details'] = extra_details
            bonds_list.append(bond_data)
            
        # Prepare Curve Points (0.0 to max_duration + 1.0)
        max_tau = float(np.max(tau))
        curve_tau = np.linspace(0.0, max_tau + 1.0, 150)
        curve_y = nelson_siegel(curve_tau, beta0, beta1, beta2, lmbda)
        
        curve_points = [{'x': float(t), 'y': float(val)} for t, val in zip(curve_tau, curve_y)]
        
        return jsonify({
            'success': True,
            'parameters': {
                'beta0': float(beta0),
                'beta1': float(beta1),
                'beta2': float(beta2),
                'lambda': float(lmbda)
            },
            'rmse': rmse,
            'bonds': bonds_list,
            'curve': curve_points
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error al procesar: {str(e)}'}), 500

if __name__ == '__main__':
    # Running on local host
    app.run(debug=True, port=5000)
