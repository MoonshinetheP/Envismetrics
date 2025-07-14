"""
CA.py - Chronoamperometry (CA) Analysis Module
----------------------------------------------

This module is part of the Envismetrics software suite and provides both visualization
and diffusion coefficient analysis for Chronoamperometry (CA) data. It reads time-dependent
potential and current measurements from electrochemical experiments and offers a standardized
approach to visualize raw signals and compute key kinetic parameters using the Cottrell equation.

Core Functions:
---------------
1. `step1()`: Raw Data Visualization
   - Generates two plots per dataset:
     - A: Applied Potential vs Time
     - B: Current vs Time
   - Useful for preliminary data inspection and quality control.

2. `step2(inter, n, a, c, x_range)`: Diffusion Coefficient Calculation
   - Applies the Cottrell equation to estimate diffusion coefficients (D) from current-time curves.
   - Performs linear regression on transformed data (Bt vs I), where Bt ~ t^(-1/2).
   - Produces:
     - Plot of raw current vs time.
     - Plot of regression used to extract D.
     - Summary table (CSV) with slope, D, and R² for each dataset.

Features:
---------
- Supports batch processing of multiple CA files.
- Configurable analysis range (`x_range`) for linear regression.
- Automatically handles file saving, figure generation, and output management.
- Saves intermediate results and metadata to `data.json` and CSV for reproducibility.

Dependencies:
-------------
- numpy, pandas, matplotlib, scipy, sklearn
- config.py
- BaseModule.py

Date: 2025  
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
import math
import re
import json
from config import *
from BaseModule import BaseModule

def get_num(filename):
    """
    Extracts the first number found in the filename using regex.

    Args:
        filename (str): File name string (e.g., '3PFOA400ppm_75075_CA.xlsx')

    Returns:
        int or None: Extracted number, or None if no match found.
    """
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None


class CA(BaseModule):
    def __init__(self, version):
        """
        Initialize the CA module with a version identifier.

        Args:
            version (str): Version name or ID for the run.
        """
        super().__init__(version)

    def step1(self):
        """
        Step 1: Load CA data files, plot potential and current over time, and save outputs.
        
        Outputs:
            - CA_form1_p1.png: Potential vs Time for all files
            - CA_form1_p2.png: Current vs Time for all files
            - data.json: Metadata and output file paths

        Returns:
            dict: Status and result metadata
        """
        data = self.read_data()
        
        # ----------- Plot Potential vs Time -----------
        for d in data:
            filename = d['filename']
            df = d['df']
            t = df['Time (s)']
            U = df['WE(1).Potential (V)']
            plt.plot(t, U, linestyle='-', linewidth=1, color='#1f77b4')

        plt.xlabel('time/s')
        plt.ylabel('Applied potential/V')
        plt.title('A', loc='left', bbox=dict(facecolor='white', edgecolor='black'))
        to_file1 = os.path.join(self.datapath, "CA_form1_p1.png")
        plt.savefig(to_file1)
        plt.close()

        # ----------- Plot Current vs Time -----------
        for d in data:
            filename = d['filename']
            df = d['df']
            t = df['Time (s)']
            I = df['WE(1).Current (A)']
            plt.scatter(t, I, s=1, c='#1f77b4')

        plt.xlabel('time/s')
        plt.ylabel('Current/A')
        plt.title('B', loc='right', bbox=dict(facecolor='white', edgecolor='black'))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        to_file2 = os.path.join(self.datapath, "CA_form1_p2.png")
        plt.savefig(to_file2)
        plt.close()

        # ----------- Save Result Metadata -----------
        data_file = os.path.join(self.datapath, 'data.json')
        if os.path.exists(data_file):
            data = json.loads(open(data_file, 'r').read())
        else:
            data = {'version': self.version}

        if 'CA' not in data:
            data['CA'] = {}

        data['CA']['form1'] = {
            'status': 'done',
            'input': {
                'uploaded_files': [],
            },
            'output': {
                'file1': to_file1.split("/")[-1],
                'file2': to_file2.split("/")[-1],
            }
        }

        with open(data_file, 'w') as f:
            f.write(json.dumps(data))
            print("saved to: {}".format(data_file))

        return {
            'status': True,
            'version': self.version,
            'message': 'Success',
            'data': data
        }
    def step2(self, inter, n, a, c, x_range=''):
        """
        Step 2: Calculate diffusion coefficient D from chronoamperometric data using Cottrell equation.

        Args:
            inter (int): Not used in current implementation.
            n (int): Number of electrons transferred in the reaction.
            a (float): Electrode surface area in cm².
            c (float): Initial analyte concentration in mol/cm³.
            x_range (str): A string specifying the range for regression in the form '[start, end]'.

        Returns:
            dict: A dictionary with image files, regression output, and calculated D values.
        """
        data = self.read_data()
        interval = len(data)

        F = 96485  # Faraday constant in C/mol
        A = a      # Electrode area from input
        C0 = c     # Initial concentration in mol/cm³

        print('electrode area (cm2):', A)

        # Parse x_range string to numeric values
        range_start, range_end = x_range.replace("[", "").replace("]", "").split(',')
        range_start = float(range_start)
        range_end = float(range_end)

        slope_set = []
        D_set = []
        R2_set = []
        to_files = []

        for d in data:
            filename = d['filename']
            df = d['df']
            j = get_num(filename)

            if j > interval:
                continue

            # Time and current data
            t = df['Time (s)'] - df['Time (s)'].iloc[0]  # Normalize time to start at 0
            I = df['WE(1).Current (A)']
            U = df['WE(1).Potential (V)']

            # Cottrell axis transformation: t^-0.5
            t_inverse_05 = t ** (-0.5)
            Bt = ((n * F * A * C0) / (math.pi ** 0.5)) * t_inverse_05  # Cottrell equation term

            # Plot time vs current
            plt.scatter(t, I, s=2, color='#1f77b4')
            plt.xlabel('Time (s)')
            plt.ylabel('Current (A)')
            plt.subplots_adjust(left=0.2)
            to_file1 = os.path.join(self.datapath, f"CA_form2_p{j}_1.png")
            plt.savefig(to_file1)
            plt.close()

            # Skip first two points to reduce instability
            Bt = Bt[2:]
            I = I[2:]

            # Perform linear regression in specified Bt range
            regression_mask = (Bt >= range_start) & (Bt <= range_end)
            Bt = Bt[regression_mask]
            I = I[regression_mask]

            slope, intercept = np.polyfit(Bt, I, 1)
            D = slope ** 2  # Diffusion coefficient approximation

            slope_set.append(slope)
            D_set.append(D)

            # Calculate R² (goodness-of-fit)
            residuals = I - (slope * Bt + intercept)
            ss_residuals = np.sum(residuals ** 2)
            ss_total = np.sum((I - np.mean(I)) ** 2)
            r_squared = 1 - (ss_residuals / ss_total)
            R2_set.append(r_squared)

            # Plot Bt vs I and regression line
            plt.scatter(Bt, I, s=2, color='#1f77b4')
            plt.plot(Bt, slope * Bt + intercept, color='red', label='Regression Line')
            plt.xlabel('nFAC₀ π ⁻¹/² t⁻¹/²')
            plt.ylabel('Current (A)')
            plt.legend()
            plt.subplots_adjust(left=0.2)
            to_file2 = os.path.join(self.datapath, f"CA_form2_p{j}_2.png")
            plt.savefig(to_file2)
            plt.close()

            print(j, "Slope:", slope)
            print(j, "R-squared:", r_squared)
            print(j, "D:", D)

            to_files.append([
                to_file1.split("/")[-1],
                to_file2.split("/")[-1],
            ])

        # Store all results in CSV
        table = pd.DataFrame([slope_set, D_set, R2_set], index=['slope', 'D', 'R2'])
        new_column_names = ['interval{}'.format(i + 2) for i in range(len(D_set))]
        table.columns = new_column_names
        to_file_csv = os.path.join(self.datapath, "CA_form2.csv")
        table.to_csv(to_file_csv, index=True, sep=',')
        print("saved to: {}".format(to_file_csv))

        data = self.res_data
        if 'CA' not in data:
            data['CA'] = {}

        data['CA']['form2'] = {
            'status': 'done',
            'input': {
                'uploaded_files': [],
            },
            'output': {
                'files': to_files,
                'csv_file': to_file_csv.split("/")[-1],
            }
        }

        self.save_result_data(data)

        return {
            'status': True,
            'version': self.version,
            'message': 'Success',
            'data': data
        }

# Test entry point (debugging)
if __name__ == '__main__':
    c = CA("version_test_CV")
