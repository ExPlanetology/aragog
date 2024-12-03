import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator, CloughTocher2DInterpolator
from scipy.ndimage import gaussian_filter

from scipy.interpolate import interp1d
from scipy import interpolate

# # #Routine for plotting the lookup tables from SPIDER.


# directory_path = './'
# units=['dT/dr','K','dT/dr',r'K$^{-1}$','K','K',r'kg/m$^{3}$','J/K',r'kg/m$^{3}$','K',r'K$^{-1}$','J/K']

# files=[]

# for filename in os.listdir(directory_path):
#     if filename.endswith(".dat"):
#         files.append(filename)
           
# # # #Translation from P-S space to P-T space through interpolation. Then projected into a regular grid.
            
# solids_files=np.array([1,2,3,6,7])
# melt_files=np.array([0,4,5,8,9])

# for i in melt_files:
#     solidus_spider_header=files[i]
#     solidus_spider=np.genfromtxt(directory_path+solidus_spider_header)
#     scaling_solidus=solidus_spider[0]
#     solidus_spider=solidus_spider[1:]*scaling_solidus

#     temp_solid_header=files[4]
#     solid=np.genfromtxt(directory_path+temp_solid_header)
#     scaling_solid=solid[0]
#     temp_solid=solid[1:]*scaling_solid
#     P2=solidus_spider[:,0][0:871]/1e9
#     da = temp_solid[0:]
#     reshaped = da.reshape(95,2020,3) # Array of shape Num_entropy, Num_pressure, temperature
#     pressure_values = reshaped[:,:,0].mean(axis=0)/1e9
#     entropy_values = reshaped[:,:,1].mean(axis=1)
#     temperature_grid = reshaped[:,:,2]

#     P_S_Quantity = solidus_spider[:, :3]  # First two columns are P, S, third is Quantity

#     # Extract the pressure and entropy from the input array
#     P = P_S_Quantity[:, 0]/1e9  # Pressure
#     S = P_S_Quantity[:, 1]  # Entropy
#     Quantity = P_S_Quantity[:, 2]  # The Quantity you want to keep unchanged


#     # Create the interpolation function with bounds checking disabled
#     interpolation_function = RegularGridInterpolator(
#         (pressure_values, entropy_values), 
#         temperature_grid.T, 
#         method='linear', 
#         bounds_error=False,  # Allow extrapolation
#         fill_value=None  # You can set a fill value for out-of-bounds cases, like np.nan or a constant
#     )
#     P_S_pairs = np.column_stack((P, S))
#     T = interpolation_function(P_S_pairs)  # Interpolate temperatures based on P, S pairs

#     # # Create new equally spaced arrays
#     T_new = np.linspace(T.min(), T.max(), num=700)
#     P_new = np.linspace(P.min(), P.max(), num=700)


#     def quantity_inter(P_row):
#         P_0 = P[np.where(np.isclose(P,P_row,atol=0.3))]
#         S_0 = S[np.where(np.isclose(P,P_row,atol=0.3))]
#         Quantity_0 = Quantity[np.where(np.isclose(P,P_row,atol=0.3))]
        
#         # Stack Pressure and Entropy pairs
#         P_S_pairs_0 = np.column_stack((P_0, S_0))
        
#         # Get temperature for P_S pairs using the interpolation function
#         T_0 = interpolation_function(P_S_pairs_0)
        
#         # Sort data by T_0 if not already sorted
#         sorted_indices = np.argsort(T_0)
#         T_0 = T_0[sorted_indices]
#         Quantity_0 = Quantity_0[sorted_indices]
        
#         # Perform 1D interpolation for Quantity based on T_0 and Quantity_0
#         interp_func = interpolate.interp1d(T_0, Quantity_0, kind='linear', bounds_error=False, fill_value=1e-15)
        
#         # Interpolate Quantity values at the positions of T_new
#         Quantity_new = interp_func(T_new)
#         return Quantity_new

#     Quantity_grid = np.vstack([quantity_inter(P_row) for P_row in P_new])
#     T_grid,P_grid = np.meshgrid(T_new,P_new)
#     P_flat = P_grid.flatten()
#     T_flat = T_grid.flatten()
#     Quantity_flat=Quantity_grid.flatten()

    # new_P_value=1e-5
    # # Find the value of Quantity where P = 0
    # index_P_zero = np.where(np.isclose(P_flat, 0, atol=1e-6))[0]
    # quantity_at_P_zero = Quantity_flat[index_P_zero][0]  # Take the first occurrence

    # # Use the value of Quantity at P = 0 as the fixed density
    # fixed_density = quantity_at_P_zero

    # # Create arrays for the new row
    # P_new_row = np.full_like(T_new, new_P_value)  # Pressure is fixed at 1e-5
    # T_new_row = T_new  # Use the existing temperature range
    # Quantity_new_row = np.full_like(T_new, fixed_density)  # Now fixed density is the Quantity at P = 0

    # # Find the index where P = 0
    # index_after_P_zero = np.argmax(P_flat > 0)  # First occurrence of P > 0 (immediately after P = 0)

    # # Insert the new row immediately after P = 0
    # P_flat_0 = np.insert(P_flat, index_after_P_zero, P_new_row)
    # T_flat_0 = np.insert(T_flat, index_after_P_zero, T_new_row)
    # Quantity_flat_0 = np.insert(Quantity_flat, index_after_P_zero, Quantity_new_row)

#     plt.figure(figsize=(7, 10))
#     plt.scatter(T_flat_0,P_flat_0, c=Quantity_flat_0, cmap='magma')
#     plt.colorbar(label=solidus_spider_header)
#     plt.ylabel('Pressure (P)')
#     plt.xlabel('Temperature (T)')
#     plt.gca().invert_yaxis()
# #         # plt.savefig(solidus_spider_header+'.png')





# #     header='#pressure temperature quantity'

# #     saved_file=np.savetxt('RG_nan_'+solidus_spider_header, X=np.array([P_flat_0*1e9,T_flat_0,Quantity_flat_0]).T,
# #             header=header,
# #             fmt='%.10e', delimiter='\t', comments='')



# # # #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import numpy as np
# import os
# from scipy.interpolate import RegularGridInterpolator, interp1d

# directory_path = './'
# units = ['dT/dr', 'K', 'dT/dr', r'K$^{-1}$', 'K', 'K', r'kg/m$^{3}$', 'J/K', r'kg/m$^{3}$', 'K', r'K$^{-1}$', 'J/K']

# files = []

# for filename in os.listdir(directory_path):
#     if filename.endswith(".dat"):
#         files.append(filename)

# # Define the consistent pressure and temperature grids for all files
# P_new = np.linspace(0, 1e9, num=700)  # Adjust this range as necessary
# T_new = np.linspace(200, 16500, num=700)  # Adjust this range as necessary

# solids_files = np.array([1, 2, 3, 6, 7])
# melt_files = np.array([0, 4, 5, 8, 9])

# def process_file(file_index,temp_file,dimension):
#     solidus_spider_header = files[file_index]
#     solidus_spider = np.genfromtxt(directory_path + solidus_spider_header)
#     scaling_solidus = solidus_spider[0]
#     solidus_spider = solidus_spider[1:] * scaling_solidus

#     temp_solid_header = files[temp_file]
#     solid = np.genfromtxt(directory_path + temp_solid_header)
#     scaling_solid = solid[0]
#     temp_solid = solid[1:] * scaling_solid
#     P2 = solidus_spider[:, 0][0:871] / 1e9
#     da = temp_solid[0:]
#     reshaped = da.reshape(dimension, 2020, 3)  # Array of shape Num_entropy, Num_pressure, temperature
#     pressure_values = reshaped[:, :, 0].mean(axis=0) / 1e9
#     entropy_values = reshaped[:, :, 1].mean(axis=1)
#     temperature_grid = reshaped[:, :, 2]

#     P_S_Quantity = solidus_spider[:, :3]  # First two columns are P, S, third is Quantity

#     # Extract the pressure and entropy from the input array
#     P = P_S_Quantity[:, 0] / 1e9  # Pressure
#     S = P_S_Quantity[:, 1]  # Entropy
#     Quantity = P_S_Quantity[:, 2]  # The Quantity you want to keep unchanged

#     # Create the interpolation function with bounds checking disabled
#     interpolation_function = RegularGridInterpolator(
#         (pressure_values, entropy_values),
#         temperature_grid.T,
#         method='linear',
#         bounds_error=False,  # Allow extrapolation
#         fill_value=None  # You can set a fill value for out-of-bounds cases, like np.nan or a constant
#     )
#     P_S_pairs = np.column_stack((P, S))
#     T = interpolation_function(P_S_pairs)  # Interpolate temperatures based on P, S pairs

#     # Create new equally spaced arrays
#     T_new = np.linspace(T.min(), T.max(), num=700)
#     P_new = np.linspace(P.min(), P.max(), num=700)

#     def quantity_inter(P_row):
#         P_0 = P[np.isclose(P, P_row, atol=0.3)]
#         S_0 = S[np.isclose(P, P_row, atol=0.3)]
#         Quantity_0 = Quantity[np.isclose(P, P_row, atol=0.3)]

#         # Stack Pressure and Entropy pairs
#         P_S_pairs_0 = np.column_stack((P_0, S_0))

#         # Get temperature for P_S pairs using the interpolation function
#         T_0 = interpolation_function(P_S_pairs_0)

#         # Sort data by T_0 if not already sorted
#         sorted_indices = np.argsort(T_0)
#         T_0 = T_0[sorted_indices]
#         Quantity_0 = Quantity_0[sorted_indices]

#         # Perform 1D interpolation for Quantity based on T_0 and Quantity_0
#         interp_func = interp1d(T_0, Quantity_0, kind='linear', bounds_error=False, fill_value=1e-15)
#         Quantity_new = interp_func(T_new)
#         return Quantity_new

#     # Create quantity grid aligned to unified P_new and T_new grids
#     Quantity_grid = np.vstack([quantity_inter(P_row) for P_row in P_new])

#     # Generate unified grids for pressure and temperature
#     T_grid, P_grid = np.meshgrid(T_new, P_new)
#     P_flat, T_flat, Quantity_flat = P_grid.flatten(), T_grid.flatten(), Quantity_grid.flatten()

#     # Insert the new row with fixed pressure and density
#     new_P_value = 1e-5
#     # Find the value of Quantity where P = 0
#     index_P_zero = np.where(np.isclose(P_flat, 0, atol=1e-6))[0]
#     quantity_at_P_zero = Quantity_flat[index_P_zero][0]  # Take the first occurrence

#     # Use the value of Quantity at P = 0 as the fixed density
#     fixed_density = quantity_at_P_zero

#     # Create arrays for the new row
#     P_new_row = np.full_like(T_new, new_P_value)  # Pressure is fixed at 1e-5
#     T_new_row = T_new  # Use the existing temperature range
#     Quantity_new_row = np.full_like(T_new, fixed_density)  # Now fixed density is the Quantity at P = 0

#     # Find the index where P = 0
#     index_after_P_zero = np.argmax(P_flat > 0)  # First occurrence of P > 0 (immediately after P = 0)

#     # Insert the new row immediately after P = 0
#     P_flat_0 = np.insert(P_flat, index_after_P_zero, P_new_row)
#     T_flat_0 = np.insert(T_flat, index_after_P_zero, T_new_row)
#     Quantity_flat_0 = np.insert(Quantity_flat, index_after_P_zero, Quantity_new_row)

#     # Save the interpolated data
#     header = '#pressure temperature quantity'
#     np.savetxt(f'RG_{solidus_spider_header}', np.array([P_flat_0, T_flat_0, Quantity_flat_0]).T, header=header, fmt='%.10e', delimiter='\t', comments='')
    
#     plt.figure(figsize=(7, 10))
#     plt.scatter(T_flat_0,P_flat_0, c=Quantity_flat_0, cmap='magma')
#     plt.colorbar(label=solidus_spider_header)
#     plt.ylabel('Pressure (P)')
#     plt.xlabel('Temperature (T)')
#     plt.gca().invert_yaxis()



# for i in solids_files:
#     process_file(i,1,125)

# for i in melt_files:
#     process_file(i,4,95)


import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator, interp1d

directory_path = './'
units = ['dT/dr', 'K', 'dT/dr', r'K$^{-1}$', 'K', 'K', r'kg/m$^{3}$', 'J/K', r'kg/m$^{3}$', 'K', r'K$^{-1}$', 'J/K']

files = []

for filename in os.listdir(directory_path):
    if filename.endswith(".dat"):
        files.append(filename)


# Pass global_P_new and global_T_new to each call to process_file
def process_file(file_index, temp_file, dimension):
    solidus_spider_header = files[file_index]
    solidus_spider = np.genfromtxt(directory_path + solidus_spider_header)
    scaling_solidus = solidus_spider[0]
    solidus_spider = solidus_spider[1:] * scaling_solidus

    temp_solid_header = files[temp_file]
    solid = np.genfromtxt(directory_path + temp_solid_header)
    scaling_solid = solid[0]
    temp_solid = solid[1:] * scaling_solid
    P2 = solidus_spider[:, 0][0:871] / 1e9
    da = temp_solid[0:]
    reshaped = da.reshape(dimension, 2020, 3)  # Array of shape Num_entropy, Num_pressure, temperature
    pressure_values = reshaped[:, :, 0].mean(axis=0) / 1e9
    entropy_values = reshaped[:, :, 1].mean(axis=1)
    temperature_grid = reshaped[:, :, 2]

    P_S_Quantity = solidus_spider[:, :3]  # First two columns are P, S, third is Quantity

    # Extract the pressure and entropy from the input array
    P = P_S_Quantity[:, 0] / 1e9  # Pressure
    S = P_S_Quantity[:, 1]  # Entropy
    Quantity = P_S_Quantity[:, 2]  # The Quantity you want to keep unchanged

    # Create the interpolation function with bounds checking disabled
    interpolation_function = RegularGridInterpolator(
        (pressure_values, entropy_values),
        temperature_grid.T,
        method='linear',
        bounds_error=False,  # Allow extrapolation
        fill_value=None  # You can set a fill value for out-of-bounds cases, like np.nan or a constant
    )
    P_S_pairs = np.column_stack((P, S))
    T = interpolation_function(P_S_pairs)  # Interpolate temperatures based on P, S pairs

    # Create new equally spaced arrays
    T_new = np.linspace(0, 16500, num=1000)
    P_new = np.linspace(0, 135,num=1000)

    def quantity_inter(P_row):
        # Adjust quantity_inter to use the global T_new for interpolation
        P_0 = P[np.isclose(P, P_row, atol=0.3)]
        S_0 = S[np.isclose(P, P_row, atol=0.3)]
        Quantity_0 = Quantity[np.isclose(P, P_row, atol=0.3)]

        # Stack Pressure and Entropy pairs
        P_S_pairs_0 = np.column_stack((P_0, S_0))

        # Get temperature for P_S pairs using the interpolation function
        T_0 = interpolation_function(P_S_pairs_0)

        # Sort data by T_0 if not already sorted
        sorted_indices = np.argsort(T_0)
        T_0 = T_0[sorted_indices]
        Quantity_0 = Quantity_0[sorted_indices]

        # Perform 1D interpolation for Quantity based on T_0 and Quantity_0
        interp_func = interp1d(T_0, Quantity_0, kind='linear', bounds_error=False, fill_value=1e-15)
        Quantity_new = interp_func(T_new)
        return Quantity_new

    # Create quantity grid aligned to unified P_new and T_new grids
    Quantity_grid = np.vstack([quantity_inter(P_row) for P_row in P_new])

    # Generate unified grids for pressure and temperature
    T_grid, P_grid = np.meshgrid(T_new, P_new)

    P_flat, T_flat, Quantity_flat = P_grid.flatten(), T_grid.flatten(), Quantity_grid.flatten()

    # # Insert the new row with fixed pressure and density
    # new_P_value = 1e-5
    # # Find the value of Quantity where P = 0
    # index_P_zero = np.where(np.isclose(P_flat, 0, atol=1e-6))[0]
    # quantity_at_P_zero = Quantity_flat[index_P_zero][0]  # Take the first occurrence

    # # Use the value of Quantity at P = 0 as the fixed density
    # fixed_density = quantity_at_P_zero

    # # Create arrays for the new row
    # P_new_row = np.full_like(T_new, new_P_value)  # Pressure is fixed at 1e-5
    # T_new_row = T_new  # Use the existing temperature range
    # Quantity_new_row = np.full_like(T_new, fixed_density)  # Now fixed density is the Quantity at P = 0

    # # Find the index where P = 0
    # index_after_P_zero = np.argmax(P_flat > 0)  # First occurrence of P > 0 (immediately after P = 0)

    # # Insert the new row immediately after P = 0
    # P_flat_0 = np.insert(P_flat, index_after_P_zero, P_new_row)
    # T_flat_0 = np.insert(T_flat, index_after_P_zero, T_new_row)
    # Quantity_flat_0 = np.insert(Quantity_flat, index_after_P_zero, Quantity_new_row)


    # Save the interpolated data
    header = '#pressure temperature quantity'
    np.savetxt(f'test_{solidus_spider_header}', np.array([P_flat_0*1e9, T_flat_0, Quantity_flat_0]).T, header=header, fmt='%.10e', delimiter='\t', comments='')
    
    plt.figure(figsize=(7, 10))
    plt.scatter(T_flat,P_flat, c=Quantity_flat, cmap='magma')
    plt.colorbar(label=solidus_spider_header)
    plt.ylabel('Pressure (P)')
    plt.xlabel('Temperature (T)')
    plt.gca().invert_yaxis()

solids_files = np.array([1, 2, 3, 6, 7])
melt_files = np.array([0, 4, 5, 8, 9])


for i in solids_files:
    process_file(i,1,125)

# for i in melt_files:
#     process_file(i,4,95)
# process_file(7,1,125)

