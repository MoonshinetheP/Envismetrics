import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.CA import *


@pytest.fixture
def sample_dataframe():
    """Create mock CA data as a DataFrame."""
    t = np.linspace(0.1, 10, 100)
    data = {
        'Time (s)': t,
        'WE(1).Current (A)': 1 / np.sqrt(t),
        'WE(1).Potential (V)': np.ones_like(t),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_data(sample_dataframe):
    """Mock return value for the read_data() method."""
    return [{'filename': '3PFOA400ppm_75075_CA.xlsx', 'df': sample_dataframe}]


@patch.object(CA, 'read_data')
def test_step1_success(mock_read_data, tmp_path, mock_data):
    """Test whether step1 runs successfully and generates plot images."""
    mock_read_data.return_value = mock_data
    ca = CA("test_version")
    ca.datapath = tmp_path

    result = ca.step1()

    assert result['status'] is True
    assert 'file1' in result['data']['CA']['form1']['output']
    assert 'file2' in result['data']['CA']['form1']['output']
    assert os.path.exists(tmp_path / result['data']['CA']['form1']['output']['file1'])
    assert os.path.exists(tmp_path / result['data']['CA']['form1']['output']['file2'])


@patch.object(CA, 'read_data')
def test_step2_success(mock_read_data, tmp_path, mock_data):
    """Test whether step2 computes regression correctly and outputs plots and CSV."""
    mock_read_data.return_value = mock_data
    ca = CA("test_version")
    ca.datapath = tmp_path
    ca.res_data = {}

    result = ca.step2(inter=0, n=1, a=0.07, c=0.000001, x_range='[0, 100]')

    assert result['status'] is True
    out = result['data']['CA']['form2']['output']
    assert 'files' in out
    assert 'csv_file' in out
    for img1, img2 in out['files']:
        assert os.path.exists(tmp_path / img1)
        assert os.path.exists(tmp_path / img2)
    assert os.path.exists(tmp_path / out['csv_file'])

@patch.object(CA, 'read_data')
def test_step1_missing_columns(mock_read_data, tmp_path):
    """Test step1 with missing columns should return an error."""
    df_missing = pd.DataFrame({
        'Time (s)': np.linspace(0, 10, 50),
        'WE(1).Current (A)': np.random.rand(50)
        # 'WE(1).Potential (V)' is missing
    })
    mock_read_data.return_value = [{'filename': 'test.xlsx', 'df': df_missing}]
    ca = CA("test_version")
    ca.datapath = tmp_path

    result = ca.step1()

    assert result['status'] is False
    assert 'Missing columns' in result['message']


@patch.object(CA, 'read_data')
def test_step2_with_empty_data(mock_read_data, tmp_path):
    """Test step2 with empty dataframe should not crash and return error."""
    empty_df = pd.DataFrame(columns=['Time (s)', 'WE(1).Current (A)', 'WE(1).Potential (V)'])
    mock_read_data.return_value = [{'filename': 'empty.xlsx', 'df': empty_df}]
    ca = CA("test_version")
    ca.datapath = tmp_path
    ca.res_data = {}

    result = ca.step2(inter=0, n=1, a=0.07, c=0.000001, x_range='[0, 1]')

    assert result['status'] is False
    assert isinstance(result['message'], str)
    assert result['message'] != ''


def test_get_num_valid_filename():
    """Test get_num returns integer when number is present in filename."""
    assert get_num('file_800rpm_CA.xlsx') == 800
    assert get_num('3PFOA400ppm_75075_CA.xlsx') == 3


def test_get_num_invalid_filename():
    """Test get_num returns None when no number is found."""
    assert get_num('no_number_here.txt') is None