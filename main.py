from flask import Flask
from flask import Blueprint, jsonify, request, send_file
import pandas as pd
import numpy as np
from flask_cors import CORS
import matplotlib.pyplot as plt
import io

# Use the 'Agg' backend for non-interactive plotting
plt.switch_backend('Agg')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to detect spikes
def count_spikes(df):
    spike_rows = []
    for i in range(1, len(df) - 1):
        if df.loc[i-1, 'Current'] < 0 and df.loc[i, 'Current'] > 0 and df.loc[i+1, 'Current'] < 0:
            spike_rows.append(i)
    return spike_rows

# Function to identify charge and discharge cycles
def identify_cycles(df, spike_rows):
    df['Cycle'] = np.nan

    cycle_number = 0
    charging = False
    discharging = False

    for i in range(1, len(df)):
        if i in spike_rows:
            continue  # Skip spike rows

        if df.loc[i, 'Current'] > 0:
            if not charging:
                charging = True
                discharging = False
                cycle_number += 1  # Start a new charging cycle
            df.loc[i, 'Cycle'] = cycle_number
        elif df.loc[i, 'Current'] < 0:
            if not discharging:
                discharging = True
                charging = False
            df.loc[i, 'Cycle'] = cycle_number
        elif df.loc[i, 'Current'] == 0:
            charging = False
            discharging = False

    # Fill NaN values forward to ensure continuity within cycles
    df['Cycle'].fillna(method='ffill', inplace=True)

    # Replace remaining NaN values with 0
    df['Cycle'].fillna(0, inplace=True)

    # Convert to integer type for clarity
    df['Cycle'] = df['Cycle'].astype(int)

    return df

# Function to get the last cell difference value and row index of each complete cycle
def get_last_cell_diff_values(df):
    last_cell_diff_values = []
    end_row_indices = []

    unique_cycles = df['Cycle'].unique()
    for cycle in unique_cycles:
        cycle_data = df[df['Cycle'] == cycle]

        # Ensure the cycle ends at discharge (negative current)
        discharge_data = cycle_data[cycle_data['Current'] < 0]

        # Get the last cell difference at the end of the discharge cycle
        if not discharge_data.empty:
            last_cell_diff = discharge_data['Cell Diff'].iloc[-1]
            end_row_index = discharge_data.index[-1]
            last_cell_diff_values.append(last_cell_diff)
            end_row_indices.append(end_row_index)

    return last_cell_diff_values, end_row_indices

# Function to count spikes and get cell differences at spikes
def get_spikes_and_diffs(df, complete_cycles):
    spike_rows = count_spikes(df)
    spikes_in_cycles = 0
    cell_diff_at_spikes = []

    unique_cycles = df['Cycle'].unique()
    for cycle in unique_cycles:
        if cycle in complete_cycles:
            cycle_data = df[df['Cycle'] == cycle]
            cycle_spike_rows = [i for i in spike_rows if i in cycle_data.index]
            spikes_in_cycles += len(cycle_spike_rows)
            cell_diff_at_spikes.extend(cycle_data.loc[cycle_spike_rows, 'Cell Diff'].tolist())

    return spikes_in_cycles, cell_diff_at_spikes

# Function to determine battery health
def determine_battery_health(eve1_diff, eve2_diff):
    if eve2_diff < eve1_diff:
        return "increased"
    elif eve2_diff > eve1_diff:
        return "decreased"
    else:
        return "constant"

# Function to process a single file
def process_combined_file(file_path):
    try:
        df = pd.read_csv(file_path)

        # Remove leading and trailing spaces from column names
        df.columns = df.columns.str.strip()
        # Drop rows where current is 0
        df_cleaned = df[df['Current'] != 0].reset_index(drop=True)

        # Detect spikes
        spike_rows = count_spikes(df_cleaned)

        # Identify cycles
        df_cycles = identify_cycles(df_cleaned, spike_rows)

        # Perform analysis
        last_cell_diff_values, end_row_indices = get_last_cell_diff_values(df_cycles)

        # Return relevant data for further processing or saving
        return {
            'file_path': file_path,
            'last_cell_diff_values': last_cell_diff_values,
            'end_row_indices': end_row_indices,
            'spike_rows': spike_rows,
            'df_cycles': df_cycles
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
    

# Function to calculate total cycles and spikes for eve1 and eve2 combined files
def calculate_total_cycles_and_spikes(eve1_file_path, eve2_file_path, output_csv_path):
    # Process eve1 file
    eve1_result = process_combined_file(eve1_file_path)
    if eve1_result is None:
        print(f"Error processing file {eve1_file_path}")
        return
    
    # Process eve2 file
    eve2_result = process_combined_file(eve2_file_path)
    if eve2_result is None:
        print(f"Error processing file {eve2_file_path}")
        return
    
    # Calculate total cycles and spikes for eve1
    eve1_total_cycles = len(eve1_result['df_cycles']['Cycle'].unique())
    eve1_total_spikes = len(eve1_result['spike_rows'])
    
    # Calculate total cycles and spikes for eve2
    eve2_total_cycles = len(eve2_result['df_cycles']['Cycle'].unique())
    eve2_total_spikes = len(eve2_result['spike_rows'])
    
    # Save results to CSV
    data = {
        'File': ['eve1', 'eve2'],
        'Total Cycles': [eve1_total_cycles, eve2_total_cycles],
        'Total Spikes': [eve1_total_spikes, eve2_total_spikes]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

# Function to save cell differences to a CSV file
def save_cell_diffs_to_csv(file_path, cell_diffs, output_file_path):
    try:
        df = pd.DataFrame({'Cycle': list(range(1, len(cell_diffs) + 1)), 'Cell Difference': cell_diffs})
        df.to_csv(output_file_path, index=False)
        print(f"Saved cell differences to {output_file_path}")
    except Exception as e:
        print(f"Error saving cell differences to {output_file_path}: {e}")

# Function to process eve1 and eve2 files and save results
def process_and_save_eve_files(eve1_file_path, eve2_file_path, output_eve1_csv_path, output_eve2_csv_path):
    eve1_cell_diffs = process_combined_file(eve1_file_path)
    if eve1_cell_diffs is not None:
        save_cell_diffs_to_csv(output_eve1_csv_path, eve1_cell_diffs, output_eve1_csv_path)

    eve2_cell_diffs = process_combined_file(eve2_file_path)
    if eve2_cell_diffs is not None:
        save_cell_diffs_to_csv(output_eve2_csv_path, eve2_cell_diffs, output_eve2_csv_path)

@app.route('/')
def home():
    return  jsonify({
            'message': 'Hello, flask'
        })
    
@app.route('/eve1', methods=['GET'])
def process_eve1():
    try:
        # Read the output CSV file
        output_csv_path = 'battery_data/output_cycles_spikes.csv'
        df = pd.read_csv(output_csv_path)

        # Filter eve1 data
        eve1_data = df[df['File'] == 'eve1']

        if eve1_data.empty:
            return jsonify({"error": "Data not found for eve1"})

        # Convert total cycles to int
        eve1_total_cycles = int(eve1_data.iloc[0]['Total Cycles'])

        return jsonify({
            'eve1_total_cycles': eve1_total_cycles
        })
    except Exception as e:
        return jsonify({"error": str(e)})
    
# Route to process eve2 file
@app.route('/eve2', methods=['GET'])
def process_eve2():
    try:
        # Read the output CSV file
        output_csv_path = 'battery_data/output_cycles_spikes.csv'
        df = pd.read_csv(output_csv_path)

        # Filter eve1 data
        eve2_data = df[df['File'] == 'eve2']

        if eve2_data.empty:
            return jsonify({"error": "Data not found for eve2"})

        # Convert total cycles to int
        eve2_total_cycles = int(eve2_data.iloc[0]['Total Cycles'])
        eve2_total_spikes = int(eve2_data.iloc[0]['Total Spikes'])

        return jsonify({
            'eve2_total_cycles': eve2_total_cycles,
            'eve2_total_spikes': eve2_total_spikes
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/eve1_cell_diff', methods=['GET'])
def get_eve1_cell_data():
    try:
        # Read the CSV file
        df = pd.read_csv('battery_data/eve1_cell_diff.csv')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.plot(df['Cycle Number'], df['Cell Difference'], marker='o', linestyle='-', color='white', markerfacecolor='white')
        ax.set_title('Cell Difference over Cycles', color='white')
        ax.set_xlabel('Cycle Number', color='white')
        ax.set_ylabel('Cell Difference', color='white')
        ax.grid(True, color='gray')
        
        # Set tick colors
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Close the plot
        plt.close()
        
        # Send the image file as a response
        return send_file(img, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/eve2_cell_diff', methods=['GET'])
def get_eve2_cell_data():
    try:
        # Read the CSV file
        df = pd.read_csv('battery_data/eve2_cell_diff.csv')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.plot(df['Cycle Number'], df['Cell Difference'], marker='o', linestyle='-', color='white', markerfacecolor='white')
        ax.set_title('Cell Difference over Cycles', color='white')
        ax.set_xlabel('Cycle Number', color='white')
        ax.set_ylabel('Cell Difference', color='white')
        ax.grid(True, color='gray')
        
        # Set tick colors
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Close the plot
        plt.close()
        
        # Send the image file as a response
        return send_file(img, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/compare', methods=['GET'])
def compare():
    try:
        # Read the CSV file
        df = pd.read_csv('battery_data/comparison_results.csv')
        
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#66b3ff')
        # Plot EVE1 and EVE2 Last Cell Differences
        ax.plot(df['Cycle Number'], df['EVE1 Last Cell Diff'], label='EVE1 Last Cell Diff', marker='o', color='#00802b')
        ax.plot(df['Cycle Number'], df['EVE2 Last Cell Diff'], label='EVE2 Last Cell Diff', marker='o', color='#d9046b')
        
        # Customize plot elements
        ax.set_xlabel('Cycle Number', color='black')
        ax.set_ylabel('Last Cell Diff', color='black')
        ax.set_title('Last Cell Diff over Cycle Numbers', color='black')
        ax.legend()
        ax.grid(True, color='white')
        
        # Set tick colors
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        
        # Customize background color
        ax.set_facecolor('#66b3ff')  # blue background
        
        # Save the plot to a BytesIO object
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        
        # Close the plot
        plt.close(fig)
        
        # Send the image file as a response
        return send_file(img, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/eve1_data', methods=['GET'])
def get_eve1_data():
    try:
        # Read CSV file
        df = pd.read_csv('battery_data/eve1_combined.csv')
        # Select the first 5 rows and the specified columns
        df = df[['Cell Diff', 'Charge MOS', 'Current', 'Date', 'Discharge MOS', 'Max Temp', 'SOC', 'Seconds', 'Voltage']].head(1000)
        # Convert dataframe to JSON
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/eve2_data', methods=['GET'])
def get_eve2_data():
    try:
        # Read CSV file
        df = pd.read_csv('battery_data/eve2_combined.csv')
        # Select the first 5 rows and the specified columns
        df = df[['Cell Diff', 'Charge MOS', 'Current', 'Date', 'Discharge MOS', 'Max Temp', 'SOC', 'Seconds', 'Voltage']].head(1000)
        # Convert dataframe to JSON
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/data_compare', methods=['GET'])
def data_compare():
    try:
        df = pd.read_csv('battery_data/comparison_results.csv')
        # Select the first 5 rows and the specified columns
        df = df[['Cycle Number', 'EVE1 Last Cell Diff','EVE2 Last Cell Diff','Battery Health','Total Spikes in EVE2','Cell Diff at Spikes in EVE2']]
        # Convert dataframe to JSON
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)
