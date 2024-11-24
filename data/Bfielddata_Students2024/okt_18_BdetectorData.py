#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:22:08 2024

@author: miko
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.optimize import curve_fit
from tabulate import tabulate


def read_csv_data(filename):
    xdata, ydata1, ydata2 = [], [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the first line

        for row in reader:
            xdata.append(float(row[0]))
            ydata1.append(float(row[1]))
            ydata2.append(float(row[3]))
            
    return xdata, ydata1, ydata2

def plot_data(xdata, ydata1, ydata2):
    fig, ax1 = plt.subplots()

    # Plot ydata1 on the left y-axis
    ax1.plot(xdata, ydata1, 'b-', marker='.', label='Y Data 1')
    ax1.set_xlabel('X Data')
    ax1.set_ylabel('Y Data 1', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(xdata, ydata2, 'r-', marker='.', label='Y Data 2')
    ax2.set_ylabel('Y Data 2', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Title and grid
    fig.suptitle('X vs Y Data 1 and Y Data 2')
    ax1.grid(True)

    plt.show()
  
    
def analyze_data(data):
    # Find amplitude of oscillations
    amplitude = (np.max(data) - np.min(data)) / 2
    
    # Detect the minimum difference between any two points, greater than zero
    diffs = np.abs(np.diff(data))
    min_diff = np.min(diffs[diffs > 0])  # Filter out zero differences
    
    return amplitude, min_diff


# Define the model function for N harmonics
def model_function(t, A, *params):
    N = (len(params) // 2)  # Number of harmonics
    f = A
    for i in range(N):
        Bi = params[i]  # B_i terms
        phi_i = params[N + i]  # phi_i terms
        
        if False:
            #cosine for plotting
            f += Bi * np.cos(2 * np.pi * 0.05 * (i + 1) * t + phi_i)
        else:
            #sine for plotting 
            f += Bi * np.sin(2 * np.pi * 0.05 * (i + 1) * t - phi_i)
        
    return f





def fit_harmonics(t_data, f_data, N=5, initial_guess=None, model_function=None, use_sine=True, plotFlag=False, printFlag=False):
    """
    Fit a model with N harmonics to the given data.
    
    Parameters:
    - t_data: array-like, time data
    - f_data: array-like, frequency data
    - N: int, number of harmonics (default is 5)
    - model_function: callable, the model function to fit
    - use_sine: bool, if True, uses sine for initial guess; otherwise, uses cosine
    
    Returns:
    - params: array, fitted parameters
    - covariance: array, covariance matrix of the fit
    """
    
    if initial_guess is None:
        # Provide an initial guess for A, B1, B2, ..., BN, phi1, phi2, ..., phiN
        if use_sine:
            initial_guess = [-0.16] + [0.2] * N + [np.pi]* N
        else:
            initial_guess = [-0.16] + [0.2] * N + [0]* N


    # Define parameter bounds
    lower_bounds = [np.min(f_data) - 1] + [0] * N + [0] * N
    upper_bounds = [np.max(f_data) + 1] + [10] * N + [2 * np.pi] * N

    # Fit the data
    params, covariance = curve_fit(model_function, t_data, f_data, p0=initial_guess, bounds=(lower_bounds, upper_bounds))

    # Extract the fitted parameters
    A_fit = params[0]
    B_fit = params[1:N + 1]
    phi_fit = params[N + 1:]
    phi_fit_shifted = phi_fit - [np.pi / 2] * N

    # Generate the fitted curve
    t_fit = np.linspace(min(t_data), max(t_data), 100)
    f_fit = model_function(t_fit, A_fit, *params[1:])
    
    if plotFlag:
        # Plot the data and the fitted model
        plt.plot(t_data, f_data, 'bo', label='Experimental data')
        plt.plot(t_fit, f_fit, 'r-', label=f'{N}th order harmonics fit')
        plt.xlabel('Offset time (ms)')
        plt.ylabel('Y-observable')
        plt.title('Fitting Model to Data')
        plt.legend()
        plt.grid(True)
        plt.show()

    if printFlag:
        # Normalize B terms and format results
        B_normalized = B_fit / np.sum(B_fit)
    
        # Prepare the table data for display
        table_data = [
            ['A', f"{A_fit:.3f}"],
            ['B terms', format_array(B_fit)],
            ['B normalized', format_array(B_normalized)],
            ['phi terms', format_array(np.rad2deg(phi_fit)) + '°'],
            ['phi terms shifted by pi/2', format_array(np.rad2deg(phi_fit_shifted)) + '°']
            ]
        
        # Print the results in a table format
        print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))

    return params, covariance

def format_array(arr):
    """Format a NumPy array with 3 decimal places."""
    return np.array2string(arr, precision=3, separator=',', floatmode='fixed', suppress_small=True)

    
# filename  = 'no_comp_1.txt' 
filename  ='test_file_data.txt'
xdata_points, ydata1, ydata2 = read_csv_data(filename)

xdata = np.linspace(0,20,len(xdata_points))

if True:
    plot_data(xdata, ydata1, ydata2)

if True:
    amp1, min_diff1 = analyze_data(ydata1)
    print(f'Data1. Ampllitude: {amp1:.5f}\n Min. diff: {min_diff1:.2e}')
    
    amp2, min_diff2 = analyze_data(ydata2)
    print(f'Data2. Ampllitude: {amp2:.5f}\n Min. diff: {min_diff2:.2e}')



if True:
    initial_guess = [4.86] + [0.2] * 3 + [np.pi]* 3

    # Assuming you have defined `model_function` already
    params, covariance = fit_harmonics(xdata, ydata2, N=3, initial_guess=initial_guess, model_function=model_function, use_sine=False, plotFlag=True, printFlag=True)


# # --- COMPARE WITH THE OTHER DATA --- 
# # Reading data points from a csv file generated from excel
# def read_csv_from_line(file_path, N, optionFlag):
#     result_dict = {'offress_cool_time_ms': [], 'frequency_aom_MHz': [], 'filenumber': []}
    
#     with open(file_path, mode='r') as file:
#         csv_reader = csv.reader(file)
#         for i, row in enumerate(csv_reader):
#             if i < N - 1:  # Skip lines before N
#                 continue
#             if optionFlag == 'even-only':
#                 if i % 2 == 0:  # Process only even-numbered lines (0-based index) 
#                     result_dict['offress_cool_time_ms'].append(row[0])
#                     result_dict['frequency_aom_MHz'].append(row[1])
#                     result_dict['filenumber'].append(row[2])
            
#             if optionFlag == 'odd-only':
#                 if not (i % 2 == 0):  # Process only even-numbered lines (0-based index) 
#                     result_dict['offress_cool_time_ms'].append(row[0])
#                     result_dict['frequency_aom_MHz'].append(row[1])
#                     result_dict['filenumber'].append(row[2])    
                    
#             if optionFlag == 'all':
#                 result_dict['offress_cool_time_ms'].append(row[0])
#                 result_dict['frequency_aom_MHz'].append(row[1])
#                 result_dict['filenumber'].append(row[2])                       
    
#     return result_dict


# def dataset(choice):
#     # This is data from the 
#     if choice == 1:
#         t_data = [3,5,7,9,11,13,15,17,19,21,23,25,27]
#         f_data = np.array([
#             73.4155, 73.4163, 73.4177, 73.4197, 73.4197,
#             73.4190, 73.4188, 73.4176, 73.4158, 73.4151,
#             73.4157, 73.4157, 73.4171 
#             ])-73.4163
    
#     if choice ==2:     
#         file_path = 'fullLinePositionData091024.csv'
#         N = 5  # Start reading from the 5th line
#         data = read_csv_from_line(file_path, N, optionFlag='all')
#         # print(data['offress_cool_time_ms'])

#         # Note that if there is not full table, clearing the empty fields may cause different number of elements. 
#         t_data = np.array([float(value) for value in data['offress_cool_time_ms'] if value != ''])
#         f_data = np.array([float(value) for value in data['frequency_aom_MHz'] if value != ''])
#         f_data = f_data - f_data[0]
        
#     return t_data, f_data

# t_data, f_data =dataset(choice=2)
# # f_data_reshuffle = f_data[]
# plt.plot(t_data-3, f_data)

# fig, ax1 = plt.subplots()

# # Plot ydata1 on the left y-axis
# ax1.plot(xdata, ydata1, 'b-', marker='.', label='Y Data 1')
# ax1.set_xlabel('X Data')
# ax1.set_ylabel('Y Data 1', color='b')
# ax1.tick_params(axis='y', labelcolor='b')

# # Create a second y-axis sharing the same x-axis
# ax2 = ax1.twinx()
# ax2.plot(t_data-3, f_data, 'r-', marker='.', label='Y Data 2')
# ax2.set_ylabel('Y Data 2', color='r')
# ax2.tick_params(axis='y', labelcolor='r')

# # Title and grid
# fig.suptitle('X vs Y Data 1 and Y Data 2')
# ax1.grid(True)

# plt.show()
