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
    Extracts the first numeric segment from a filename.

    Parameters:
        filename (str): The name of the file.

    Returns:
        int or None: The extracted integer, or None if no number found.
    """
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None


class CA(BaseModule):
    """
    Chronoamperometry (CA) module for visualizing time vs. current/potential
    and for calculating diffusion coefficient using Cottrell analysis.
    """

    def __init__(self, version):
        """
        Initialize CA module.

        Parameters:
            version (str): Version tag or label for current analysis session.
        """
        super().__init__(version)

    def step1(self):
        """
        Step 1: Generate two plots for uploaded CA data.
        - Plot A: Time vs. Potential (U)
        - Plot B: Time vs. Current (I)

        Returns:
            dict: Status and output filenames for plots.
        """
        data = self.read_data()

        # First plot: Time vs Potential
        for d in data:
            df = d['df']
            plt.plot(df['Time (s)'], df['WE(1).Potential (V)'], linestyle='-', linewidth=1, color='#1f77b4')

        plt.xlabel('time/s')
        plt.ylabel('Applied potential/V')
        plt.title('A', loc='left', bbox=dict(facecolor='white', edgecolor='black'))
        to_file1 = os.path.join(self.datapath, "CA_form1_p1.png")
        plt.savefig(to_file1)
        plt.close()

        # Second plot: Time vs Current
        for d in data:
            df = d['df']
            plt.scatter(df['Time (s)'], df['WE(1).Current (A)'], s=1, c='#1f77b4')

        plt.xlabel('time/s')
        plt.ylabel('Current/A')
        plt.title('B', loc='right', bbox=dict(facecolor='white', edgecolor='black'))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        to_file2 = os.path.join(self.datapath, "CA_form1_p2.png")
        plt.savefig(to_file2)
        plt.close()

        # Write results to data.json
        data_file = os.path.join(self.datapath, 'data.json')
        if os.path.exists(data_file):
            data = json.loads(open(data_file, 'r').read())
        else:
            data = {'version': self.version}
        if 'CA' not in data:
            data['CA'] = {}

        data['CA']['form1'] = {
            'status': 'done',
            'input': {'uploaded_files': []},
            'output': {
                'file1': os.path.basename(to_file1),
                'file2': os.path.basename(to_file2),
            }
        }

        with open(data_file, 'w') as f:
            json.dump(data, f)
            print("saved to:", data_file)

        return {
            'status': True,
            'version': self.version,
            'message': 'Success',
            'data': data
        }

    def step2(self, inter, n, a, c, x_range=''):
        """
        Step 2: Perform Cottrell fitting to determine diffusion coefficient (D)
        using linear fit of I vs t^{-0.5} (after normalization).

        Parameters:
            inter (int): Number of expected intervals (ignored in current logic).
            n (int): Number of electrons transferred.
            a (float): Electrode area in cm².
            c (float): Analyte concentration in mol/cm³.
            x_range (str): Fitting range for Bt in form '[start,end]'.

        Returns:
            dict: Result dictionary with plot/image files and CSV export.
        """
        data = self.read_data()
        interval = len(data)
        F = 96485  # Faraday constant
        A = a      # Electrode area
        C0 = c     # Concentration

        range_start, range_end = map(float, x_range.strip("[]").split(','))
        slope_set, D_set, R2_set, to_files = [], [], [], []

        for d in data:
            filename = d['filename']
            df = d['df']
            j = get_num(filename)
            if j > interval:
                continue

            # Normalize time
            t = df['Time (s)'] - df['Time (s)'].iloc[0]
            I = df['WE(1).Current (A)']
            Bt = ((n * F * A * C0) / (math.pi ** 0.5)) * (t ** -0.5)

            # Save raw t–I plot
            plt.scatter(t, I, s=2, color='#1f77b4')
            plt.xlabel('Time (s)')
            plt.ylabel('Current (A)')
            to_file1 = os.path.join(self.datapath, f"CA_form2_p{j}_1.png")
            plt.savefig(to_file1)
            plt.close()

            # Linear fit within specified Bt range
            Bt = Bt[2:]
            I = I[2:]
            mask = (Bt >= range_start) & (Bt <= range_end)
            Bt, I = Bt[mask], I[mask]

            slope, intercept = np.polyfit(Bt, I, 1)
            residuals = I - (slope * Bt + intercept)
            r_squared = 1 - np.sum(residuals ** 2) / np.sum((I - np.mean(I)) ** 2)

            D = slope ** 2
            slope_set.append(slope)
            D_set.append(D)
            R2_set.append(r_squared)

            # Plot Bt–I with regression line
            plt.scatter(Bt, I, s=2, color='#1f77b4')
            plt.plot(Bt, slope * Bt + intercept, color='red', label='Regression Line')
            plt.xlabel('nFAC₀ π⁻¹/² t⁻¹/²')
            plt.ylabel('Current (A)')
            plt.legend()
            to_file2 = os.path.join(self.datapath, f"CA_form2_p{j}_2.png")
            plt.savefig(to_file2)
            plt.close()

            print(j, "Slope:", slope, "R²:", r_squared, "D:", D)
            to_files.append([os.path.basename(to_file1), os.path.basename(to_file2)])

        # Export summary table
        table = pd.DataFrame([slope_set, D_set, R2_set], index=['slope', 'D', 'R2'])
        table.columns = [f'interval{i + 2}' for i in range(len(D_set))]
        to_file_csv = os.path.join(self.datapath, "CA_form2.csv")
        table.to_csv(to_file_csv)
        print("saved to:", to_file_csv)

        # Save to result json
        data = self.res_data
        if 'CA' not in data:
            data['CA'] = {}

        data['CA']['form2'] = {
            'status': 'done',
            'input': {'uploaded_files': []},
            'output': {
                'files': to_files,
                'csv_file': os.path.basename(to_file_csv)
            }
        }
        self.save_result_data(data)

        return {
            'status': True,
            'version': self.version,
            'message': 'Success',
            'data': data
        }

if __name__ == '__main__':
    c = CA("version_test_CV")
