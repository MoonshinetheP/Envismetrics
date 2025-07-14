import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json
from config import *
from BaseModule import BaseModule

class CV(BaseModule):
    def __init__(self, version):
        super().__init__(version)

    def step1(self):
        """
        Step 1: Plot current vs potential for multiple CV files and save the figure.
        """
        data = self.read_data()
        for d in data:
            filename = d['filename']
            df = d['df']
            x = df['WE(1).Potential (V)']
            y = df['WE(1).Current (A)']
            plt.plot(x, y, linewidth=1, label=filename)

        plt.xlabel('Potential (V)')
        plt.ylabel('Current (A)')
        plt.title('Cyclic Voltammetry')
        plt.legend(fontsize=6)
        plt.tight_layout()

        to_file = os.path.join(self.datapath, "CV_form1.png")
        plt.savefig(to_file)
        plt.close()

        # Save metadata
        data_file = os.path.join(self.datapath, 'data.json')
        data = json.loads(open(data_file, 'r').read()) if os.path.exists(data_file) else {'version': self.version}

        data['CV'] = data.get('CV', {})
        data['CV']['form1'] = {
            'status': 'done',
            'input': {'uploaded_files': []},
            'output': {'file': to_file.split("/")[-1]}
        }
        with open(data_file, 'w') as f:
            f.write(json.dumps(data))
            print("saved to:", data_file)

        return {
            'status': True,
            'version': self.version,
            'message': 'Success',
            'data': data
        }

    def step2(self):
        """
        Step 2: Randles–Ševčík analysis to estimate diffusion coefficient D from CV peaks.
        """
        data = self.read_data()
        result_data = []

        for d in data:
            filename = d['filename']
            df = d['df']

            v = self._get_scanrate(filename)  # in mV/s
            v = v / 1000  # convert to V/s

            I = df['WE(1).Current (A)']
            ip = max(abs(I))  # peak current
            result_data.append((v ** 0.5, ip))

        result_data = np.array(result_data)
        x = result_data[:, 0].reshape(-1, 1)
        y = result_data[:, 1].reshape(-1, 1)

        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        r2 = model.score(x, y)

        # Randles-Sevcik equation:
        # ip = (2.69e5) * n^(3/2) * A * D^(1/2) * C * v^(1/2)
        # Solve for D
        n, A, C = 1, 0.07, 1e-6  # Example values: n=1 e-, A=0.07 cm2, C=1e-6 mol/cm3
        D = (slope / (2.69e5 * n**1.5 * A * C))**2

        # Plot
        plt.scatter(x, y, label='Data')
        plt.plot(x, model.predict(x), color='red', label='Fit')
        plt.xlabel('√v (V^0.5/s^0.5)')
        plt.ylabel('ip (A)')
        plt.legend()
        plt.title('Randles–Ševčík Analysis')

        to_file = os.path.join(self.datapath, "CV_form2.png")
        plt.savefig(to_file)
        plt.close()

        data = self.res_data
        data['CV'] = data.get('CV', {})
        data['CV']['form2'] = {
            'status': 'done',
            'input': {},
            'output': {
                'csv_file': '',
                'slope': slope,
                'D': D,
                'R2': r2,
                'plot_file': to_file.split("/")[-1]
            }
        }
        self.save_result_data(data)

        return {
            'status': True,
            'version': self.version,
            'message': 'Success',
            'data': data
        }

    def _get_scanrate(self, filename):
        """
        Extract scan rate from filename using regex. Expects format like 'sample_100mVs_CV.xlsx'.
        """
        match = re.search(r'(\d+)mVs', filename)
        return int(match.group(1)) if match else 100  # default to 100 mV/s
