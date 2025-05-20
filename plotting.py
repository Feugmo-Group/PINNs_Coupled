import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

mesh = pv.read("outputs/model/constraints/interior.vtp")

# Check what arrays are available
print("Available data arrays:", mesh.array_names)

t_values = mesh["t"]
c_values = mesh["c"]
c__tvalues = mesh["c__t"]

print(t_values.shape)
print(c_values.shape)
print(c__tvalues.shape)

print(c__tvalues)