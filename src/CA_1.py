# ================================================================
# ✨ Updates in step2() – 2025-07 by Huize
#
# 1. ✅ Added visual distinction for regression vs. non-regression points:
#    - Points used in linear regression (Bt vs I) are shown in blue.
#    - Points excluded from the fit are shown in gray.
#    - Red line shows the linear regression result.
#
# 2. ✅ Enabled Advanced Mode support for per-transient parameter input:
#    - Now supports 'n', 'a', 'c' as either:
#        • single float (applied to all transients), or
#        • list of values (e.g., c = [1e-3, 5e-4, 2e-3]) matching the number of uploaded files.
#    - Also supports JSON-style string input: e.g., "[0.001, 0.0005, 0.002]"
#    - Automatically validates length and applies corresponding values per transient.
#
# 3. 🧩 All updates are backward-compatible with previous interface and parameters.
# ================================================================


import os as os
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
    # 定义要匹配的字符串
    # filename = "(3)PFOA400ppm_75075_50s_CA.xlsx"
    # filename = "3PFOA400ppm_75075_50s_CA.xlsx"

    # 使用正则表达式匹配数字部分
    match = re.search(r'(\d+)', filename)

    # 如果匹配成功，则打印括号中的内容
    if match:
        result = match.group(1)
        return int(result)
    else:
        return None


class CA(BaseModule):
    def __init__(self, version):
        super().__init__(version)

    def step1(self):
        data = self.read_data()
        for d in data:
            filename = d['filename']
            df = d['df']

            t = df['Time (s)']
            I = df['WE(1).Current (A)']
            U = df['WE(1).Potential (V)']
            plt.plot(t, U, linestyle='-', linewidth=1, color='#1f77b4')

        plt.xlabel('time/s')
        plt.ylabel('Applied potential/V')
        plt.title('A', loc='left', bbox=dict(facecolor='white', edgecolor='black'))
        # plt.ylim(-2e-5,2e-5)
        # plt.grid()
        # plt.show()
        to_file1 = os.path.join(self.datapath, "CA_form1_p1.png")
        plt.savefig(to_file1)
        plt.close()


        for d in data:
            filename = d['filename']
            df = d['df']

            t = df['Time (s)']
            I = df['WE(1).Current (A)']
            U = df['WE(1).Potential (V)']

            plt.scatter(t, I, s=1, c='#1f77b4')

        plt.xlabel('time/s')
        plt.ylabel('Current/A')
        plt.title('B', loc='right', bbox=dict(facecolor='white', edgecolor='black'))
        # plt.ylim(-2e-5,2e-5)
        # plt.grid()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        to_file2 = os.path.join(self.datapath, "CA_form1_p2.png")
        plt.savefig(to_file2)
        plt.close()

        data_file = os.path.join(self.datapath, 'data.json')
        if os.path.exists(data_file):
            data = json.loads(open(data_file, 'r').read())
        else:
            data = {'version': self.version}

        if 'CA' not in data.keys():
            data['CA'] = {}

        data['CA']['form1'] = {
            'status': 'done',
            'input': {
                'uploaded_files': [],
                # 'sigma': self.sigma
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
        data = self.read_data()
        interval = len(data)
        num_files = len(data)
        

        F = 96485

        # 支持单值或列表传入
        def parse_param(p, name):
            if isinstance(p, (int, float)):
                return [p] * num_files
            elif isinstance(p, list):
                assert len(p) == num_files, f"Length of {name} list must match number of uploaded files ({num_files})"
                return p
            elif isinstance(p, str) and p.startswith("["):
                parsed = json.loads(p)
                assert len(parsed) == num_files, f"Length of {name} list must match number of uploaded files ({num_files})"
                return parsed
            else:
                raise ValueError(f"Invalid {name} parameter: must be float or list of length {num_files}")

        # 支持 advanced 模式（列表），否则是统一值
        n_list = parse_param(n, 'n')
        a_list = parse_param(a, 'a')
        c_list = parse_param(c, 'c')

        range_start, range_end = x_range.replace("[", "").replace("]", "").split(',')
        range_start = float(range_start)
        range_end = float(range_end)

        slope_set = []
        D_set = []
        R2_set = []
        to_files = []
        
        for idx, d in enumerate(data):
            filename = d['filename']
            df = d['df']
            j = get_num(filename)

            if j > interval:
                continue   
                
            # 当前 transient 使用各自参数
            n_i = n_list[idx]
            A = a_list[idx]
            C0 = c_list[idx]

            t = df['Time (s)'] - df['Time (s)'].iloc[0]
            I = df['WE(1).Current (A)']

            # 画原始 t vs I 图
            plt.scatter(t, I, s=2, color='#1f77b4')
            plt.xlabel('Time (s)')
            plt.ylabel('Current (A)')
            plt.subplots_adjust(left=0.2)
            to_file1 = os.path.join(self.datapath, f"CA_form2_p{j}_1.png")
            plt.savefig(to_file1)
            plt.close()

            # 构造 Bt vs I 数据
            Bt_full = ((n_i * F * A * C0) / (math.pi ** 0.5)) * (t ** -0.5)
            I_full = I

            Bt_valid = Bt_full[2:]
            I_valid = I_full[2:]

            regression_mask = (Bt_valid >= range_start) & (Bt_valid <= range_end)
            Bt_reg = Bt_valid[regression_mask]
            I_reg = I_valid[regression_mask]
            Bt_nonreg = Bt_valid[~regression_mask]
            I_nonreg = I_valid[~regression_mask]

            # 线性拟合 Bt vs I（回归区间）
            slope, intercept = np.polyfit(Bt_reg, I_reg, 1)
            D = slope ** 2

            slope_set.append(slope)
            D_set.append(D)

            residuals = I_reg - (slope * Bt_reg + intercept)
            ss_residuals = np.sum(residuals ** 2)
            ss_total = np.sum((I_reg - np.mean(I_reg)) ** 2)
            r_squared = 1 - (ss_residuals / ss_total)
            R2_set.append(r_squared)

            # 画 Bt vs I 图
            plt.scatter(Bt_nonreg, I_nonreg, s=2, color='#bbbbbb', label='Excluded')
            plt.scatter(Bt_reg, I_reg, s=2, color='#1f77b4', label='Fitted')
            plt.plot(Bt_reg, slope * Bt_reg + intercept, color='red', label='Regression Line')
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
            to_files.append([to_file1.split("/")[-1], to_file2.split("/")[-1]])

        table = pd.DataFrame([slope_set, D_set, R2_set], index=['slope', 'D', 'R2'])
        new_column_names = [f'interval{i + 2}' for i in range(len(D_set))]
        table.columns = new_column_names
        to_file_csv = os.path.join(self.datapath, "CA_form2.csv")
        table.to_csv(to_file_csv, index=True, sep=',')
        print("saved to: {}".format(to_file_csv))

        data = self.res_data
        if 'CA' not in data.keys():
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




if __name__ == '__main__':
    c = CA("version_test_CV")


