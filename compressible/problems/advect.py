from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg
import math

def init_data(my_data, rp):
    """ initialize the advect problem """

    msg.bold("initializing the advect problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in advect.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:,:] = 1.0
    xmom[:,:] = 0.0
    ymom[:,:] = 0.0

    r_init = rp.get_param("advect.r_init")
    u = rp.get_param("advect.u")
    v = rp.get_param("advect.v")

    gamma = rp.get_param("eos.gamma")
    pi = math.pi

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    myg = my_data.grid

    r = np.sqrt((myg.x2d - xctr)**2 + (myg.y2d - yctr)**2)
    
    dens[r < r_init] = 2.0

    # pressure is constant
    pres = myg.scratch_array()
    pres[:,:] = 1.0

    # find the energy
    rhoe = pres[:,:]/(gamma - 1.0)

    # velocity
    xmom[:,:] = dens[:,:]*u
    ymom[:,:] = dens[:,:]*v
    
    ener[:,:] = rhoe + 0.5*(xmom**2 + ymom**2)/dens


def finalize():
    """ print out any information to the user at the end of the run """

    pass
