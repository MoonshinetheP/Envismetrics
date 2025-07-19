# test_HDV.py

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from HDV import HDV

@pytest.fixture
def mock_HDV(tmp_path):
    """
    Create a mock HDV instance with necessary paths and empty result data.
    """
    version = "test_version"
    hdv = HDV(version)
    hdv.datapath = str(tmp_path)  # Temporary path for saving plots
    hdv.files_info = str(tmp_path / "mock_info.json")
    hdv.res_data = {}
    return hdv

@pytest.fixture
def mock_dataframe():
    """
    Create a mock dataframe with necessary columns.
    """
    df = pd.DataFrame({
        'WE(1).Current (A)': np.random.normal(0, 1e-4, 100),
        'WE(1).Potential (V)': np.linspace(-1.0, 0.5, 100),
        'Potential applied (V)': np.linspace(-1.0, 0.5, 100)
    })
    return df

@patch.object(HDV, 'read_data')
def test_step1_success(mock_read_data, mock_HDV, mock_dataframe):
    """
    Test HDV.step1() with mock data and verify success status and output files.
    """
    # Mock the read_data method to return fake data with RPM keys
    mock_read_data.return_value = {
        '400rpm': mock_dataframe,
        '800rpm': mock_dataframe,
        '1200rpm': mock_dataframe
    }

    # Run step1
    result = mock_HDV.step1(sigma=5)

    # Assertions
    assert result['status'] is True
    assert result['message'] == 'Success'
    assert 'file1' in result['data']['HDV']['form1']['output']
    assert 'file2' in result['data']['HDV']['form1']['output']

    # Check if plot files were created
    file1 = os.path.join(mock_HDV.datapath, result['data']['HDV']['form1']['output']['file1'])
    file2 = os.path.join(mock_HDV.datapath, result['data']['HDV']['form1']['output']['file2'])
    assert os.path.exists(file1)
    assert os.path.exists(file2)

@pytest.fixture
def mock_HDV_step2(tmp_path):
    """
    Create a mock HDV instance for step2_1 testing.
    """
    hdv = HDV("test_version")
    hdv.datapath = str(tmp_path)
    hdv.files_info = str(tmp_path / "mock_info.json")
    hdv.res_data = {
        "HDV": {
            "form1": {
                "input": {
                    "sigma": 5.0
                }
            }
        }
    }
    return hdv

@pytest.fixture
def mock_step2_data():
    """
    Create mock data for multiple RPMs.
    """
    E = np.linspace(-1.0, 0.5, 100)
    I = np.random.normal(0, 1e-4, 100)
    df = pd.DataFrame({
        'WE(1).Potential (V)': E,
        'WE(1).Current (A)': I,
        'Potential applied (V)': E,
    })
    return {
        '400rpm': df.copy(),
        '800rpm': df.copy(),
        '1600rpm': df.copy()
    }

@pytest.fixture
def step2_all_params():
    """
    Define all necessary parameters for step2_1.
    """
    return {
        'input_N': 1,
        'input_A': 0.07,
        'input_V': 0.01,
        'input_C': 1e-6,
        'input_range': '(-0.9, 0.4)',
        'input_n_points': 5,
        'input_interval': 10
    }

@patch.object(HDV, 'read_data')
def test_step2_1_success(mock_read_data, mock_HDV_step2, mock_step2_data, step2_all_params):
    """
    Test HDV.step2_1 with mock data and parameter inputs.
    """
    # Patch the data returned by read_data
    mock_read_data.return_value = mock_step2_data

    # Call step2_1 with all parameters
    result = mock_HDV_step2.step2_1(step2_all_params)

    # Check response structure and status
    assert result['status'] is True
    assert result['message'] == 'Success'
    assert 'file1' in result['data']['HDV']['form2_1']['output']
    assert 'file2' in result['data']['HDV']['form2_1']['output']
    assert 'excel_file' in result['data']['HDV']['form2_1']['output']

    # Check files exist
    file1 = os.path.join(mock_HDV_step2.datapath, result['data']['HDV']['form2_1']['output']['file1'])
    file2 = os.path.join(mock_HDV_step2.datapath, result['data']['HDV']['form2_1']['output']['file2'])
    excel_file = os.path.join(mock_HDV_step2.datapath, result['data']['HDV']['form2_1']['output']['excel_file'])

    assert os.path.exists(file1)
    assert os.path.exists(file2)
    assert os.path.exists(excel_file)

# test_HDV_step2_2.py

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch
from HDV import HDV

@pytest.fixture
def mock_HDV_step2_2(tmp_path):
    """
    Create a mock HDV instance for step2_2 testing.
    """
    hdv = HDV("test_version")
    hdv.datapath = str(tmp_path)
    hdv.files_info = str(tmp_path / "mock_info.json")
    hdv.res_data = {
        "HDV": {
            "form1": {
                "input": {
                    "sigma": 5.0
                }
            }
        }
    }
    return hdv

@pytest.fixture
def mock_step2_2_data():
    """
    Create mock data for multiple RPMs with required columns.
    """
    E = np.linspace(-1.0, 0.5, 100)
    I = np.random.normal(0, 1e-4, 100)
    df = pd.DataFrame({
        'Potential applied (V)': E,
        'WE(1).Current (A)': I,
        'WE(1).Potential (V)': E
    })
    return {
        '400rpm': df.copy(),
        '800rpm': df.copy(),
        '1600rpm': df.copy()
    }

@pytest.fixture
def step2_2_all_params():
    """
    Define all necessary parameters for step2_2.
    """
    return {
        'input_N': 1,
        'input_A': 0.07,
        'input_V': 0.01,
        'input_C': 1e-6,
        'input_range': '(-0.9, 0.4)',
        'input_n_points': 5,
        'input_interval': 10
    }

@patch.object(HDV, 'read_data')
def test_step2_2_success(mock_read_data, mock_HDV_step2_2, mock_step2_2_data, step2_2_all_params):
    """
    Test HDV.step2_2 with mock data and parameter inputs.
    """
    # Patch the read_data method to return mock data
    mock_read_data.return_value = mock_step2_2_data

    # Run step2_2
    result = mock_HDV_step2_2.step2_2(step2_2_all_params)

    # Check response structure and status
    assert result['status'] is True
    assert 'file1' in result['data']['HDV']['form2_2']['output']
    assert 'file2' in result['data']['HDV']['form2_2']['output']

    # Check files exist
    file1 = os.path.join(mock_HDV_step2_2.datapath, result['data']['HDV']['form2_2']['output']['file1'])
    file2 = os.path.join(mock_HDV_step2_2.datapath, result['data']['HDV']['form2_2']['output']['file2'])

    assert os.path.exists(file1)
    assert os.path.exists(file2)

