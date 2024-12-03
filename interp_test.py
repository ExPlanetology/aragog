
from aragog.phase import LookupProperty2D
import numpy as np
import matplotlib.pyplot as plt

# Path to the solidus and melt files
solidus_file = 'data/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018/solidus_monteux.dat'
melt_file = 'data/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018/liquidus_monteux.dat'

# Load the data from the solidus file into a NumPy array
solidus_data = np.loadtxt(solidus_file)
melt_data = np.loadtxt(melt_file)

# Now, solidus_data and melt_data are NumPy arrays, so we can access their columns
# Assuming the first column is pressure and the second column is temperature
pressure_solidus = solidus_data[:, 0]  # First column is pressure
temperature_solidus = solidus_data[:, 1]  # Second column is temperature

pressure_melt = melt_data[:, 0]  # First column is pressure
temperature_melt = melt_data[:, 1]  # Second column is temperature

# Load lookup data for solid and melt densities
solid_density_data = np.loadtxt("/Users/marianity/Desktop/Solidus_depression/lookup_data/aragog/data/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018/density_solid.dat")
melt_density_data = np.loadtxt("/Users/marianity/Desktop/Solidus_depression/lookup_data/aragog/data/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018/density_melt.dat")

# Initialize the LookupProperty2D with the lookup data
lookup_solid = LookupProperty2D(name="Solid Density from File", value=solid_density_data)
lookup_melt = LookupProperty2D(name="Melt Density from File", value=melt_density_data)

# Evaluate the property at the given pressure and temperature for solidus
result_solidus = lookup_solid.eval(temperature_solidus, pressure_solidus)

# Evaluate the property at the given pressure and temperature for melt
result_melt = lookup_melt.eval(temperature_melt, pressure_melt)








# -------------------------------------------------------------------------------------------------------------------------------
from scipy.interpolate import griddata


# Create an interpolator for solidus using solid density data
points_density_solid = np.column_stack((solid_density_data[:,0], solid_density_data[:,1]))
densities_interpolated_solidus = griddata(points_density_solid,solid_density_data[:,2] , 
                                          (pressure_solidus, temperature_solidus), 
                                          method='nearest')

# Create an interpolator for melt using melt density data
points_density_melt = np.column_stack((melt_density_data[:,0], melt_density_data[:,1]))
densities_interpolated_melt = griddata(points_density_melt, melt_density_data[:,2], 
                                       (pressure_melt, temperature_melt), 
                                       method='nearest')

# Handle extrapolated points (set to NaN if outside convex hull)
densities_interpolated_solidus[np.isnan(densities_interpolated_solidus)] = np.nan
densities_interpolated_melt[np.isnan(densities_interpolated_melt)] = np.nan

print(f"Solidus - Pressure: {pressure_solidus}, Temperature: {temperature_solidus}, Density Solid from Aragog: {result_solidus},Corresponding Density: {densities_interpolated_solidus}")
print(f"Melt - Pressure: {pressure_melt}, Temperature: {temperature_melt},  Density Melt from Aragog: {result_melt},Corresponding Density: {densities_interpolated_melt}")


# Create the plot plot for density difference
plt.figure(figsize=(8, 10))
plt.scatter(densities_interpolated_solidus,pressure_solidus/1e9, label='Density solidus with griddata',color='blue')
plt.scatter(densities_interpolated_melt, pressure_melt/1e9, label='Density melt with griddata',color='red')
plt.plot(result_solidus, pressure_solidus/1e9,label='Density solidus with RectBivariateSpline',color='green')
plt.plot(result_melt,pressure_melt/1e9,label='Density melt with RectBivariateSpline',color='purple')
plt.xlabel('Density (kg/m^3)')
plt.ylabel('Pressure (GPa)')
plt.title('Density comparison between RectBiVariateSpline and griddata')
plt.legend()
plt.gca().invert_yaxis()
plt.show()


# -------------------------------------------------------------------------------------------------------------------------------

# Using RectBivariateSpline

from scipy.interpolate import RectBivariateSpline
import numpy as np

# Ensure your data is on a regular grid for RectBivariateSpline
def prepare_data_for_spline(data):
    # Extract x, y, and z values
    x_values = np.unique(data[:, 0])  # Unique pressure values
    y_values = np.unique(data[:, 1])  # Unique temperature values
    
    # Create a grid for z values
    z_values = data[:, 2].reshape((x_values.size, y_values.size), order='F')  # Ensure correct ordering
    return x_values, y_values, z_values

# Prepare solid density data
x_solid, y_solid, z_solid = prepare_data_for_spline(solid_density_data)
solid_spline = RectBivariateSpline(x_solid, y_solid, z_solid,s=0)

# Prepare melt density data
x_melt, y_melt, z_melt = prepare_data_for_spline(melt_density_data)
melt_spline = RectBivariateSpline(x_melt, y_melt, z_melt)

# Evaluate densities using the splines
densities_interpolated_solidus = solid_spline(pressure_solidus, temperature_solidus, grid=False)
densities_interpolated_melt = melt_spline(pressure_melt, temperature_melt, grid=False)

# Handle extrapolated points (RectBivariateSpline extrapolates; add NaNs if required manually)
densities_interpolated_solidus[np.isnan(densities_interpolated_solidus)] = np.nan
densities_interpolated_melt[np.isnan(densities_interpolated_melt)] = np.nan

# Mimic griddata's behavior (set out-of-bounds values to NaN)
mask = (pressure_solidus >= x_solid.min()) & (pressure_solidus <= x_solid.max()) & \
       (temperature_solidus >= y_solid.min()) & (temperature_solidus <= y_solid.max())

densities_interpolated_solidus[~mask] = np.nan


# Print results
print(f"Solidus - Pressure: {pressure_solidus}, Temperature: {temperature_solidus}, "
      f" Corresponding Density: {densities_interpolated_solidus}")
print(f"Melt - Pressure: {pressure_melt}, Temperature: {temperature_melt}, "
      f" Corresponding Density: {densities_interpolated_melt}")

# Create the plot plot for density difference
plt.figure(figsize=(8, 10))
plt.plot(densities_interpolated_solidus,pressure_solidus/1e9, label='Density solidus with griddata',color='blue')
plt.plot(densities_interpolated_melt, pressure_melt/1e9, label='Density melt with griddata',color='red')
plt.xlabel('Density (kg/m^3)')
plt.ylabel('Pressure (GPa)')
plt.title('Density comparison between RectBiVariateSpline and griddata')
plt.legend()
plt.gca().invert_yaxis()
plt.show()




# -------------------------------------------------------------------------------------------------------------------------------
from scipy.interpolate import RectBivariateSpline, griddata
import numpy as np
import matplotlib.pyplot as plt


# Ensure your data is on a regular grid for RectBivariateSpline
def prepare_data_for_spline(data):
    # Extract x, y, and z values
    x_values = np.unique(data[:, 0])  # Unique pressure values
    y_values = np.unique(data[:, 1])  # Unique temperature values
    
    # Create a grid for z values
    z_values = np.full((x_values.size, y_values.size), np.nan)
    
    # Find the indices of the x and y values in the unique arrays
    x_indices = np.searchsorted(x_values, data[:, 0])
    y_indices = np.searchsorted(y_values, data[:, 1])
    
    # Fill the z_values grid
    z_values[x_indices, y_indices] = data[:, 2]
    
    return x_values, y_values, z_values
# Prepare solid density data
x_solid, y_solid, z_solid = prepare_data_for_spline(solid_density_data)
solid_spline = RectBivariateSpline(x_solid, y_solid, z_solid, s=0)

# Prepare melt density data
x_melt, y_melt, z_melt = prepare_data_for_spline(melt_density_data)
melt_spline = RectBivariateSpline(x_melt, y_melt, z_melt, s=0)

# Evaluate densities using the splines
densities_interpolated_solidus_spline = solid_spline(pressure_solidus, temperature_solidus, grid=False)
densities_interpolated_melt_spline = melt_spline(pressure_melt, temperature_melt, grid=False)


# Handle extrapolated points (RectBivariateSpline extrapolates; add NaNs if required manually)
mask_solidus = (pressure_solidus >= x_solid.min()) & (pressure_solidus <= x_solid.max()) & \
               (temperature_solidus >= y_solid.min()) & (temperature_solidus <= y_solid.max())
densities_interpolated_solidus_spline[~mask_solidus] = np.nan

mask_melt = (pressure_melt >= x_melt.min()) & (pressure_melt <= x_melt.max()) & \
            (temperature_melt >= y_melt.min()) & (temperature_melt <= y_melt.max())
densities_interpolated_melt_spline[~mask_melt] = np.nan


# Create the plot for density comparison
plt.figure(figsize=(10, 6))
plt.plot(densities_interpolated_solidus_spline, pressure_solidus / 1e9, label='Density Solidus (Spline)', color='blue')
plt.plot(densities_interpolated_melt_spline, pressure_melt / 1e9, label='Density Melt (Spline)', color='red')
plt.xlabel('Density (kg/m^3)')
plt.ylabel('Pressure (GPa)')
plt.title('Density comparison between RectBiVariateSpline and griddata')
plt.legend()
plt.gca().invert_yaxis()
plt.show()
