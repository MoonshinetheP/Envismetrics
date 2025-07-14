"""
HDV.py - Hydrodynamic Voltammetry (HDV) Analysis Module
--------------------------------------------------------

This module is part of the Envismetrics software suite and performs comprehensive 
analysis of Hydrodynamic Voltammetry (HDV) data, including both Levich and 
Koutecky-Levich plots for determining diffusion coefficients.

Main Features:
--------------
1. **step1(sigma)**:
   - Loads all valid HDV data files (Excel or TXT), sorted by RPM.
   - Visualizes raw and Gaussian-filtered current vs potential curves.
   - Stores PNG figures for raw and smoothed voltammograms.

2. **step2_1(all_params)**:
   - Levich plot analysis for each potential point or interval.
   - Calculates slope (B) and diffusion coefficient (D).
   - Produces:
     - Individual regression plots across rotation rates.
     - Summary slope-D plots vs potential.
     - Exportable CSV file with current vs √ω data.

3. **step2_2(all_params)**:
   - Koutecky-Levich plot analysis for kinetic and diffusion current separation.
   - Estimates diffusion coefficients using inverse current vs ω⁻¹ᐟ².
   - Produces:
     - Regression line plots per potential.
     - Combined B-D dual-axis plots.
     - Exportable CSV of calculated parameters.

Core Utilities:
---------------
- `read_data()`: Handles file parsing, RPM extraction, and CSV caching.
- `rpm_to_rads()`: Converts RPM to angular velocity (rad/s).
- Utility functions: `find_y`, `extract_rpm`, `check_files`, etc.

Inputs:
-------
- User-defined electrochemical parameters: number of electrons (n), area (A), viscosity (ν), and bulk concentration (C).
- Potential range, number of points, and sampling interval.

Output:
-------
- PNG plots, CSV results, and progress tracking in structured JSON.

Dependencies:
-------------
- numpy, pandas, matplotlib, scipy, sklearn
- BaseModule.py
- JSON configuration files

Date: 2025  
"""
