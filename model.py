import os
import warnings
import math
import scipy.special
import torch
import numpy as np
from sympy import Symbol, Eq, exp, sin, sqrt, pi, erfc
import scipy

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry import Bounds
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.utils.sympy.functions import parabola
from physicsnemo.sym.key import Key
from physicsnemo.sym.utils.vpinn.test_functions import(
    RBF_Function,
    Test_Function,
    Legendre_test,
    Trig_test,
)
from physicsnemo.sym.geometry.primitives_1d import Point1D
from physicsnemo.sym.utils.vpinn.integral import tensor_int, Quad_Rect, Quad_Collection
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    VariationalConstraint,
)
from physicsnemo.sym.dataset import DictVariationalDataset
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.node import Node
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from physicsnemo.sym.loss import Loss
from sympy import Symbol, Heaviside, Eq
import quadpy
from physicsnemo.sym.geometry import Parameter, Parameterization
from eqns_pdes import PNP

from physicsnemo.sym.utils.io.vtk import VTKUniformGrid
from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    
    # make nodes to unroll graph on
    pnp = PNP()
    pnp_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys = [Key("u"), Key("c")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = pnp.make_nodes()+[pnp_net.make_node(name="pnp_network")]

    ## add constraints and make geometry
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")

    L = ((0,0), (1,1)) # Length of box
    time = (0,8)
    geo = Rectangle((0,0), (1,1))
    time_range = {t: time}

    domain = Domain()

    # initial conditions
    #IC = PointwiseBoundaryConstraint(
    #    nodes = nodes,
    #    geometry = geo,
    #    outvar = {"u": 1},
    #    batch_size=cfg.batch_size.IC, ##NEED TO ADD TO CONFIG
    #    parameterization={t:0},
    #)
    #domain.add_constraint(IC, "IC")

    #boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = geo,
        outvar={"c": 1, "u":0},
        criteria=~Eq(x, 0),
        batch_size=cfg.batch_size.BC, 
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")

    #boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = geo,
        outvar={"c": 2, "u": 10},
        criteria=~Eq(x, 1),
        batch_size=cfg.batch_size.BC, 
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")

    #boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = geo,
        outvar={"u__y":0,"c__y":0},
        criteria=~Eq(y, 1),
        batch_size=cfg.batch_size.BC, 
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")    

    #boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = geo,
        outvar={"u__y":0,"c__y":0},
        criteria=~Eq(y, 0),
        batch_size=cfg.batch_size.BC, 
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")    

    # interior
    interior = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry = geo,
        outvar={"poisson": 0, "NernstP": 0},
        batch_size = cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")
    
    vtk_obj = VTKUniformGrid(
    bounds=[[0, 1], [0, 1]],  # Your box bounds
    npoints=[100, 100],  # Grid resolution in each dimension
    export_map={"u": ["u"],"c":["c"]},  # Map output 'u' to 'potential' in VTK file
    )

    deltaT = 0.001
    t_max = 10
    t = np.arange(0, t_max, deltaT)
    t = np.expand_dims(t, axis=-1)
    invar_numpy = {"t": t}    
    
    potential_inferencer = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x":"x","y":"y"},
        output_names=["u","c"],
        invar=invar_numpy,
        requires_grad=True,
        batch_size=cfg.batch_size.inference,
    )    
    domain.add_inferencer(potential_inferencer)

    ##make solver
    slv = Solver(cfg, domain)

    ## start solver
    slv.solve()

if __name__ == "__main__":
    run()