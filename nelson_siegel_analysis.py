import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar

# ---------------------------------------------------------
# 1. Nelson-Siegel Model Functions
# ---------------------------------------------------------

def nelson_siegel(tau, beta0, beta1, beta2, lmbda):
    """
    Computes the Nelson-Siegel yield for a given maturity tau.
    
    Formula:
    y(tau) = beta0 + beta1 * ((1 - e^(-tau/lambda)) / (tau/lambda))
                   + beta2 * ((1 - e^(-tau/lambda)) / (tau/lambda) - e^(-tau/lambda))
    """
    tau = np.array(tau, dtype=float)
    if lmbda <= 0:
        return np.zeros_like(tau) + 1e10  # Penalty for non-positive lambda
        
    with np.errstate(divide='ignore', invalid='ignore'):
        # Limit of (1 - e^-x)/x as x -> 0 is 1.0
        factor = (1.0 - np.exp(-tau / lmbda)) / (tau / lmbda)
        factor = np.where(tau == 0, 1.0, factor)
        
        term1 = factor
        term2 = factor - np.exp(-tau / lmbda)
        
    return beta0 + beta1 * term1 + beta2 * term2

def fit_nelson_siegel(tau, y):
    """
    Fits the Nelson-Siegel model parameters (beta0, beta1, beta2, lmbda)
    to the given maturities (tau) and yields (y).
    
    Uses a hybrid approach:
    1. Grid search / scalar optimization over lambda.
    2. For each lambda, solve for beta coefficients using Ordinary Least Squares (OLS).
    3. Refine all parameters using a 4D bounded optimization to ensure stability.
    """
    tau = np.array(tau, dtype=float)
    y = np.array(y, dtype=float)
    
    # 1. Scalar optimization for lambda
    def get_beta_and_rss(lmbda_val):
        if lmbda_val <= 0.01:
            return None, 1e10
            
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = (1.0 - np.exp(-tau / lmbda_val)) / (tau / lmbda_val)
            factor = np.where(tau == 0, 1.0, factor)
            term1 = factor
            term2 = factor - np.exp(-tau / lmbda_val)
            
        X = np.column_stack([np.ones_like(tau), term1, term2])
        
        # OLS solution
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

    # Optimize lambda in a typical range [0.1, 15.0]
    res_scalar = minimize_scalar(rss_objective, bounds=(0.1, 15.0), method='bounded')
    best_lmbda = res_scalar.x
    best_beta, _ = get_beta_and_rss(best_lmbda)
    
    if best_beta is None:
        # Fallback to simple guesses
        best_beta = [np.mean(y), 0.0, 0.0]
        best_lmbda = 2.0

    # 2. Refine using 4D bounded optimization with SLSQP to enforce non-negative short-term yield (beta0 + beta1 >= 0)
    bounds = [
        (0.0, 50.0),       # beta0 (0% to 50%)
        (-50.0, 50.0),     # beta1
        (-50.0, 50.0),     # beta2
        (0.1, 20.0)        # lambda (scale/decay parameter)
    ]
    
    # Enforce beta0 + beta1 >= 0 (short rate is non-negative)
    constraints = [{'type': 'ineq', 'fun': lambda params: params[0] + params[1]}]
    
    initial_guess = [best_beta[0], best_beta[1], best_beta[2], best_lmbda]
    
    # Adjust initial guess if it violates the constraint
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
# 2. Data Cleaning and Parsing
# ---------------------------------------------------------

def clean_value(val):
    """
    Cleans a string representation of a number to a clean float.
    Handles currency symbols ($), percentage signs (%), and Spanish locale decimals (commas).
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
        
    val_str = str(val).strip()
    is_percent = False
    
    if val_str.endswith('%'):
        is_percent = True
        val_str = val_str[:-1].strip()
        
    # Remove currency signs and other spaces
    val_str = val_str.replace('$', '').replace(' ', '')
    
    # Handle Spanish decimal and thousands separator
    # Examples: "1.100,50" -> "1100.50", "108,70" -> "108.70"
    if ',' in val_str:
        if '.' in val_str:
            val_str = val_str.replace('.', '')
        val_str = val_str.replace(',', '.')
        
    try:
        num = float(val_str)
        # Keep yield on percentage scale (e.g. 6.83 instead of 0.0683) for the fit
        return num
    except ValueError:
        return np.nan

def load_and_parse_data(filepath):
    """
    Loads data from filepath and parses column names and numeric values.
    Supports tab-separated (from Excel paste), comma-separated, and semicolon-separated formats.
    """
    print(f"[*] Reading data from '{filepath}'...")
    
    # Try different separators
    separators = ['\t', ',', ';']
    df = None
    for sep in separators:
        try:
            temp_df = pd.read_csv(filepath, sep=sep)
            if temp_df.shape[1] > 1:
                df = temp_df
                print(f"[-] Successfully parsed file using '{repr(sep)}' separator.")
                break
        except Exception:
            continue
            
    if df is None:
        raise ValueError("Could not parse the file. Please ensure it is tab-separated, comma-separated, or semicolon-separated.")
        
    # Clean column headers
    df.columns = df.columns.str.strip()
    
    # Find relevant columns using pattern matching
    col_bono = None
    col_duration = None
    col_tir = None
    
    # 1. Ticker / Bond Name Column
    bono_patterns = ['bono', 'ticker', 'asset', 'activo', 'nombre', 'empresa']
    for p in bono_patterns:
        for col in df.columns:
            if p in col.lower():
                col_bono = col
                break
        if col_bono:
            break
            
    # 2. Modified Duration Column (tau)
    dur_patterns = ['m duration', 'mduration', 'modified duration', 'duracion mod', 'duration', 'duracion', 'tau']
    for p in dur_patterns:
        for col in df.columns:
            if p in col.lower():
                col_duration = col
                break
        if col_duration:
            break
            
    # 3. TIR (Yield) Column (y)
    tir_patterns = ['% tir', 'tir', 'yield', 'tea', 'ytm', 'y']
    for p in tir_patterns:
        for col in df.columns:
            if p in col.lower():
                col_tir = col
                break
        if col_tir:
            break
            
    # Fallback to defaults if column not found
    if not col_bono:
        col_bono = df.columns[1] if df.shape[1] > 1 else df.columns[0]
    if not col_duration:
        col_duration = df.columns[7] if df.shape[1] > 7 else df.columns[0]
    if not col_tir:
        col_tir = df.columns[5] if df.shape[1] > 5 else df.columns[0]
        
    print(f"[-] Auto-detected columns:")
    print(f"    - Bond Identifier: '{col_bono}'")
    print(f"    - Modified Duration (tau): '{col_duration}'")
    print(f"    - Yield / TIR (y): '{col_tir}'")
    
    # Create clean working DataFrame
    working_df = pd.DataFrame()
    working_df['Bono'] = df[col_bono].astype(str).str.strip()
    working_df['M_Duration'] = df[col_duration].apply(clean_value)
    working_df['TIR'] = df[col_tir].apply(clean_value)
    
    # Keep other columns for output details
    for col in df.columns:
        if col not in [col_bono, col_duration, col_tir]:
            working_df[col] = df[col]
            
    # Drop rows where Duration or TIR is NaN
    initial_rows = len(working_df)
    working_df = working_df.dropna(subset=['M_Duration', 'TIR']).reset_index(drop=True)
    dropped = initial_rows - len(working_df)
    if dropped > 0:
        print(f"[!] Warning: Dropped {dropped} rows with invalid/empty numeric data.")
        
    # Scale correction: check if yield is in decimals (e.g. 0.0683) instead of percentage (e.g. 6.83)
    # If the average yield is very small (< 0.2), assume decimal format and multiply by 100
    mean_yield = working_df['TIR'].mean()
    if mean_yield < 0.20:
        print(f"[*] Yield values appear to be in decimal form (mean = {mean_yield:.4f}). Converting to percentage scale.")
        working_df['TIR'] = working_df['TIR'] * 100.0
        
    return working_df

# ---------------------------------------------------------
# 3. Main Program Execution
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("      NELSON-SIEGEL YIELD CURVE & Z-SCORE ANALYSIS      ")
    print("=" * 60)
    
    input_file = "input_data.txt"
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file '{input_file}' not found.")
        print("Please create 'input_data.txt' and paste your Excel/spreadsheet data there.")
        return
        
    # Load and parse
    try:
        df = load_and_parse_data(input_file)
    except Exception as e:
        print(f"[ERROR] Loading data failed: {e}")
        return
        
    if len(df) < 4:
        print("[ERROR] Nelson-Siegel curve fitting requires at least 4 bonds. Only found", len(df))
        return
        
    tau = df['M_Duration'].values
    y = df['TIR'].values
    
    # Fit Nelson-Siegel Curve
    print("\n[*] Calibrating Nelson-Siegel Curve parameters...")
    beta0, beta1, beta2, lmbda = fit_nelson_siegel(tau, y)
    
    print("\n[-] Calibrated Parameters:")
    print(f"    - beta0 (Long-term level):      {beta0:8.4f}%")
    print(f"    - beta1 (Short-term slope):     {beta1:8.4f}%")
    print(f"    - beta2 (Medium-term curve):    {beta2:8.4f}%")
    print(f"    - lambda (Decay parameter):     {lmbda:8.4f}")
    
    # Calculate fitted yields and residuals
    df['Fitted_TIR'] = nelson_siegel(tau, beta0, beta1, beta2, lmbda)
    df['Residual'] = df['TIR'] - df['Fitted_TIR']
    df['Residual_bps'] = df['Residual'] * 100.0  # Convert to basis points (1% = 100 bps)
    
    # Compute Z-score on residuals
    # Z = (Residual - mean) / std
    mean_res = df['Residual'].mean()
    std_res = df['Residual'].std(ddof=1) if len(df) > 1 else 1.0
    
    # Handle edge case where std is very close to 0
    if std_res < 1e-6:
        std_res = 1.0
        
    df['Z_Score'] = (df['Residual'] - mean_res) / std_res
    
    # Assign recommendations based on Z-score
    # High positive Z-score = Bond yields much more than the curve (cheap / buy signal)
    # High negative Z-score = Bond yields much less than the curve (rich / sell signal)
    def get_signal(z):
        if z > 1.5:
            return "Cheap (BUY)"
        elif z > 0.5:
            return "Slightly Cheap"
        elif z < -1.5:
            return "Rich (SELL)"
        elif z < -0.5:
            return "Slightly Rich"
        else:
            return "Neutral"
            
    df['Signal'] = df['Z_Score'].apply(get_signal)
    
    # Sort by Z-score in descending order (from cheapest to richest)
    df_sorted = df.sort_values(by='Z_Score', ascending=False)
    
    print("\n" + "=" * 88)
    print(f"{'Bono':<10} | {'MDuration':<10} | {'Actual TIR':<10} | {'Fitted TIR':<10} | {'Residual (bps)':<16} | {'Z-Score':<8} | {'Signal':<15}")
    print("-" * 88)
    for idx, row in df_sorted.iterrows():
        print(f"{row['Bono']:<10} | {row['M_Duration']:<10.2f} | {row['TIR']:<9.2f}% | {row['Fitted_TIR']:<9.2f}% | {row['Residual_bps']:+14.1f} bps | {row['Z_Score']:+8.2f} | {row['Signal']:<15}")
    print("=" * 88)
    
    # Export results to CSV
    output_file = "output_data.csv"
    try:
        df_sorted.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig') # Excel-friendly semicolon delimiter
        print(f"\n[-] Saved detailed analysis to '{output_file}'.")
    except PermissionError:
        print(f"\n[ERROR] Permission denied: '{output_file}' could not be written.")
        print("        Please make sure the file is NOT open in Excel or another application, then run the script again.")
    
    # ---------------------------------------------------------
    # 4. Plotting & Visualisation
    # ---------------------------------------------------------
    print("\n[*] Generating Yield Curve Plot...")
    
    plt.figure(figsize=(11, 7))
    
    # Plot fitted curve from 0 to max(tau) + 1.0
    tau_grid = np.linspace(0.0, max(tau) + 1.0, 200)
    y_fitted_grid = nelson_siegel(tau_grid, beta0, beta1, beta2, lmbda)
    plt.plot(tau_grid, y_fitted_grid, color='#1f77b4', linestyle='-', linewidth=2.5, label='Nelson-Siegel Yield Curve')
    
    # Scatter plot of bonds, color-coded by Z-score
    # We use coolwarm colormap (Red = Rich/Low Yield, Blue = Cheap/High Yield)
    # Let's map Z-scores: negative values (rich) -> Red, positive values (cheap) -> Blue/Green
    sc = plt.scatter(tau, y, c=df['Z_Score'], cmap='coolwarm_r', s=120, edgecolor='black', zorder=5, label='Bonds (color by Z-Score)')
    cbar = plt.colorbar(sc)
    cbar.set_label('Z-Score (Positive = Cheap/BUY, Negative = Rich/SELL)', fontsize=11, fontweight='bold', labelpad=10)
    
    # Label each point with the bond ticker
    for i, txt in enumerate(df['Bono']):
        plt.annotate(
            txt, 
            (tau[i], y[i]), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center', 
            fontsize=9, 
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3, ec="gray")
        )
        
    plt.title('Curva Nelson-Siegel & Análisis de Z-Score sobre Activos', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Modified Duration (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Yield / TIR (TEA %)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', frameon=True, shadow=True)
    
    # Style details
    plt.gca().set_facecolor('#f8f9fa')
    plt.tight_layout()
    
    # Save image
    plot_file = "nelson_siegel_curve.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[-] Saved chart image to '{plot_file}'.")
    print("\n[*] Analysis completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
