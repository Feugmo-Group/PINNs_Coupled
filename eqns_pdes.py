"""
Coupled PNP equations

reference: Poisson, Nernst, Planck, Gouws, Runze et al.

"""

from sympy import Symbol, Function, Number, sin, cos, pi
from sympy.vector import divergence
from physicsnemo.sym.eq.pde import PDE

class PNP(PDE):

    name = "PNPEquation"

    def __init__(self):
        #coordinates
        x = Symbol("x") 
        y = Symbol("y")
        t = Symbol("t")

        #make input variables
        input_variables = {"x": x, "y": y, "t": t}

        #make function
        u = Function("u")(*input_variables)
        c = Function("c")(*input_variables)

        #set equations
        self.equations = {}
        self.equations["poisson"] = u.diff(x,2) + u.diff(y,2)
        self.equations["NernstP"] = (c.diff(x,2)+c.diff(y,2)) + c*(u.diff(x,2)+u.diff(y,2)) + (u.diff(x)*c.diff(x)) + (u.diff(y)*c.diff(y))

