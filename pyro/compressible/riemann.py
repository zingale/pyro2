import numpy as np
from numba import njit


@njit(cache=True)
def riemann_cgf(idir, ng,
                idens, ixmom, iymom, iener, irhoX, nspec,
                gamma, U_l, U_r, is_solid=False):
    r"""
    Solve riemann shock tube problem for a general equation of
    state using the method of Colella, Glaz, and Ferguson.  See
    Almgren et al. 2010 (the CASTRO paper) for details.

    The Riemann problem for the Euler's equation produces 4 regions,
    separated by the three characteristics (u - cs, u, u + cs)::


           u - cs    t    u      u + cs
             \       ^   .       /
              \  *L  |   . *R   /
               \     |  .     /
                \    |  .    /
            L    \   | .   /    R
                  \  | .  /
                   \ |. /
                    \|./
           ----------+----------------> x

    We care about the solution on the axis.  The basic idea is to use
    estimates of the wave speeds to figure out which region we are in,
    and: use jump conditions to evaluate the state there.

    Only density jumps across the u characteristic.  All primitive
    variables jump across the other two.  Special attention is needed
    if a rarefaction spans the axis.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    ng : int
        The number of ghost cells
    nspec : int
        The number of species
    idens, ixmom, iymom, iener, irhoX : int
        The indices of the density, x-momentum, y-momentum, internal energy density
        and species partial densities in the conserved state vector.
    gamma : float
        Adiabatic index
    U_l, U_r : ndarray
        Conserved state on the left and right cell edges.
    is_solid : bool
        Are we at a solid wall?

    Returns
    -------
    out : ndarray
        Conserved flux
    """

    nvar = U_l.shape[0]

    F = np.zeros((nvar))

    smallc = 1.e-10
    smallrho = 1.e-10
    smallp = 1.e-10

    # primitive variable states
    rho_l = U_l[idens]

    # un = normal velocity; ut = transverse velocity
    if idir == 1:
        un_l = U_l[ixmom] / rho_l
        ut_l = U_l[iymom] / rho_l
    else:
        un_l = U_l[iymom] / rho_l
        ut_l = U_l[ixmom] / rho_l

    rhoe_l = U_l[iener] - 0.5 * rho_l * (un_l**2 + ut_l**2)

    p_l = rhoe_l * (gamma - 1.0)
    p_l = max(p_l, smallp)

    rho_r = U_r[idens]

    if idir == 1:
        un_r = U_r[ixmom] / rho_r
        ut_r = U_r[iymom] / rho_r
    else:
        un_r = U_r[iymom] / rho_r
        ut_r = U_r[ixmom] / rho_r

    rhoe_r = U_r[iener] - 0.5 * rho_r * (un_r**2 + ut_r**2)

    p_r = rhoe_r * (gamma - 1.0)
    p_r = max(p_r, smallp)

    # define the Lagrangian sound speed
    W_l = max(smallrho * smallc, np.sqrt(gamma * p_l * rho_l))
    W_r = max(smallrho * smallc, np.sqrt(gamma * p_r * rho_r))

    # and the regular sound speeds
    c_l = max(smallc, np.sqrt(gamma * p_l / rho_l))
    c_r = max(smallc, np.sqrt(gamma * p_r / rho_r))

    # define the star states
    pstar = (W_l * p_r + W_r * p_l + W_l *
             W_r * (un_l - un_r)) / (W_l + W_r)
    pstar = max(pstar, smallp)
    ustar = (W_l * un_l + W_r * un_r + (p_l - p_r)) / (W_l + W_r)

    # now compute the remaining state to the left and right
    # of the contact (in the star region)
    rhostar_l = rho_l + (pstar - p_l) / c_l**2
    rhostar_r = rho_r + (pstar - p_r) / c_r**2

    rhoestar_l = rhoe_l + \
        (pstar - p_l) * (rhoe_l / rho_l + p_l / rho_l) / c_l**2
    rhoestar_r = rhoe_r + \
        (pstar - p_r) * (rhoe_r / rho_r + p_r / rho_r) / c_r**2

    cstar_l = max(smallc, np.sqrt(gamma * pstar / rhostar_l))
    cstar_r = max(smallc, np.sqrt(gamma * pstar / rhostar_r))

    # figure out which state we are in, based on the location of
    # the waves
    if ustar > 0.0:

        # contact is moving to the right, we need to understand
        # the L and *L states

        # Note: transverse velocity only jumps across contact
        ut_state = ut_l

        # define eigenvalues
        lambda_l = un_l - c_l
        lambdastar_l = ustar - cstar_l

        if pstar > p_l:
            # the wave is a shock -- find the shock speed
            sigma = (lambda_l + lambdastar_l) / 2.0

            if sigma > 0.0:
                # shock is moving to the right -- solution is L state
                rho_state = rho_l
                un_state = un_l
                p_state = p_l
                rhoe_state = rhoe_l

            else:
                # solution is *L state
                rho_state = rhostar_l
                un_state = ustar
                p_state = pstar
                rhoe_state = rhoestar_l

        else:
            # the wave is a rarefaction
            if lambda_l < 0.0 and lambdastar_l < 0.0:
                # rarefaction fan is moving to the left -- solution is
                # *L state
                rho_state = rhostar_l
                un_state = ustar
                p_state = pstar
                rhoe_state = rhoestar_l

            elif lambda_l > 0.0 and lambdastar_l > 0.0:
                # rarefaction fan is moving to the right -- solution is
                # L state
                rho_state = rho_l
                un_state = un_l
                p_state = p_l
                rhoe_state = rhoe_l

            else:
                # rarefaction spans x/t = 0 -- interpolate
                alpha = lambda_l / (lambda_l - lambdastar_l)

                rho_state = alpha * rhostar_l + (1.0 - alpha) * rho_l
                un_state = alpha * ustar + (1.0 - alpha) * un_l
                p_state = alpha * pstar + (1.0 - alpha) * p_l
                rhoe_state = alpha * rhoestar_l + \
                    (1.0 - alpha) * rhoe_l

    elif ustar < 0:

        # contact moving left, we need to understand the R and *R
        # states

        # Note: transverse velocity only jumps across contact
        ut_state = ut_r

        # define eigenvalues
        lambda_r = un_r + c_r
        lambdastar_r = ustar + cstar_r

        if pstar > p_r:
            # the wave if a shock -- find the shock speed
            sigma = (lambda_r + lambdastar_r) / 2.0

            if sigma > 0.0:
                # shock is moving to the right -- solution is *R state
                rho_state = rhostar_r
                un_state = ustar
                p_state = pstar
                rhoe_state = rhoestar_r

            else:
                # solution is R state
                rho_state = rho_r
                un_state = un_r
                p_state = p_r
                rhoe_state = rhoe_r

        else:
            # the wave is a rarefaction
            if lambda_r < 0.0 and lambdastar_r < 0.0:
                # rarefaction fan is moving to the left -- solution is
                # R state
                rho_state = rho_r
                un_state = un_r
                p_state = p_r
                rhoe_state = rhoe_r

            elif lambda_r > 0.0 and lambdastar_r > 0.0:
                # rarefaction fan is moving to the right -- solution is
                # *R state
                rho_state = rhostar_r
                un_state = ustar
                p_state = pstar
                rhoe_state = rhoestar_r

            else:
                # rarefaction spans x/t = 0 -- interpolate
                alpha = lambda_r / (lambda_r - lambdastar_r)

                rho_state = alpha * rhostar_r + (1.0 - alpha) * rho_r
                un_state = alpha * ustar + (1.0 - alpha) * un_r
                p_state = alpha * pstar + (1.0 - alpha) * p_r
                rhoe_state = alpha * rhoestar_r + \
                    (1.0 - alpha) * rhoe_r

    else:  # ustar == 0

        rho_state = 0.5 * (rhostar_l + rhostar_r)
        un_state = ustar
        ut_state = 0.5 * (ut_l + ut_r)
        p_state = pstar
        rhoe_state = 0.5 * (rhoestar_l + rhoestar_r)

    # species now
    if nspec > 0:
        if ustar > 0.0:
            xn = U_l[irhoX:irhoX + nspec] / U_l[idens]

        elif ustar < 0.0:
            xn = U_r[irhoX:irhoX + nspec] / U_r[idens]
        else:
            xn = 0.5 * (U_l[irhoX:irhoX + nspec] / U_l[idens] +
                        U_r[irhoX:irhoX + nspec] / U_r[idens])

    # are we on a solid boundary?
    if is_solid:
        un_state = 0.0

    # compute the fluxes
    F[idens] = rho_state * un_state

    if idir == 1:
        F[ixmom] = rho_state * un_state**2 + p_state
        F[iymom] = rho_state * ut_state * un_state
    else:
        F[ixmom] = rho_state * ut_state * un_state
        F[iymom] = rho_state * un_state**2 + p_state

    F[iener] = rhoe_state * un_state + \
        0.5 * rho_state * (un_state**2 + ut_state**2) * un_state + \
        p_state * un_state

    if nspec > 0:
        F[irhoX:irhoX + nspec] = xn * rho_state * un_state

    return F


@njit(cache=True)
def riemann_prim(idir, ng,
                 irho, iu, iv, ip, iX, nspec,
                 gamma, q_l, q_r, is_solid=False):
    r"""
    this is like riemann_cgf, except that it works on a primitive
    variable input state and returns the primitive variable interface
    state

    Solve riemann shock tube problem for a general equation of
    state using the method of Colella, Glaz, and Ferguson.  See
    Almgren et al. 2010 (the CASTRO paper) for details.

    The Riemann problem for the Euler's equation produces 4 regions,
    separated by the three characteristics :math:`(u - cs, u, u + cs)`::


           u - cs    t    u      u + cs
             \       ^   .       /
              \  *L  |   . *R   /
               \     |  .     /
                \    |  .    /
            L    \   | .   /    R
                  \  | .  /
                   \ |. /
                    \|./
           ----------+----------------> x

    We care about the solution on the axis.  The basic idea is to use
    estimates of the wave speeds to figure out which region we are in,
    and: use jump conditions to evaluate the state there.

    Only density jumps across the :math:`u` characteristic.  All primitive
    variables jump across the other two.  Special attention is needed
    if a rarefaction spans the axis.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    ng : int
        The number of ghost cells
    nspec : int
        The number of species
    irho, iu, iv, ip, iX : int
        The indices of the density, x-velocity, y-velocity, pressure and species fractions in the state vector.
    lower_solid, upper_solid : int
        Are we at lower or upper solid boundaries?
    gamma : float
        Adiabatic index
    q_l, q_r : ndarray
        Primitive state on the left and right cell edges.

    Returns
    -------
    out : ndarray
        Primitive flux
    """

    nvar = q_l.shape[0]

    q_int = np.zeros((nvar))

    smallc = 1.e-10
    smallrho = 1.e-10
    smallp = 1.e-10

    # primitive variable states
    rho_l = q_l[irho]

    # un = normal velocity; ut = transverse velocity
    if idir == 1:
        un_l = q_l[iu]
        ut_l = q_l[iv]
    else:
        un_l = q_l[iv]
        ut_l = q_l[iu]

    p_l = q_l[ip]
    p_l = max(p_l, smallp)

    rho_r = q_r[irho]

    if idir == 1:
        un_r = q_r[iu]
        ut_r = q_r[iv]
    else:
        un_r = q_r[iv]
        ut_r = q_r[iu]

    p_r = q_r[ip]
    p_r = max(p_r, smallp)

    # define the Lagrangian sound speed
    rho_l = max(smallrho, rho_l)
    rho_r = max(smallrho, rho_r)
    W_l = max(smallrho * smallc, np.sqrt(gamma * p_l * rho_l))
    W_r = max(smallrho * smallc, np.sqrt(gamma * p_r * rho_r))

    # and the regular sound speeds
    c_l = max(smallc, np.sqrt(gamma * p_l / rho_l))
    c_r = max(smallc, np.sqrt(gamma * p_r / rho_r))

    # define the star states
    pstar = (W_l * p_r + W_r * p_l + W_l *
             W_r * (un_l - un_r)) / (W_l + W_r)
    pstar = max(pstar, smallp)
    ustar = (W_l * un_l + W_r * un_r + (p_l - p_r)) / (W_l + W_r)

    # now compute the remaining state to the left and right
    # of the contact (in the star region)
    rhostar_l = rho_l + (pstar - p_l) / c_l**2
    rhostar_r = rho_r + (pstar - p_r) / c_r**2

    cstar_l = max(smallc, np.sqrt(gamma * pstar / rhostar_l))
    cstar_r = max(smallc, np.sqrt(gamma * pstar / rhostar_r))

    # figure out which state we are in, based on the location of
    # the waves
    if ustar > 0.0:

        # contact is moving to the right, we need to understand
        # the L and *L states

        # Note: transverse velocity only jumps across contact
        ut_state = ut_l

        # define eigenvalues
        lambda_l = un_l - c_l
        lambdastar_l = ustar - cstar_l

        if pstar > p_l:
            # the wave is a shock -- find the shock speed
            sigma = (lambda_l + lambdastar_l) / 2.0

            if sigma > 0.0:
                # shock is moving to the right -- solution is L state
                rho_state = rho_l
                un_state = un_l
                p_state = p_l

            else:
                # solution is *L state
                rho_state = rhostar_l
                un_state = ustar
                p_state = pstar

        else:
            # the wave is a rarefaction
            if lambda_l < 0.0 and lambdastar_l < 0.0:
                # rarefaction fan is moving to the left -- solution is
                # *L state
                rho_state = rhostar_l
                un_state = ustar
                p_state = pstar

            elif lambda_l > 0.0 and lambdastar_l > 0.0:
                # rarefaction fan is moving to the right -- solution is
                # L state
                rho_state = rho_l
                un_state = un_l
                p_state = p_l

            else:
                # rarefaction spans x/t = 0 -- interpolate
                alpha = lambda_l / (lambda_l - lambdastar_l)

                rho_state = alpha * rhostar_l + (1.0 - alpha) * rho_l
                un_state = alpha * ustar + (1.0 - alpha) * un_l
                p_state = alpha * pstar + (1.0 - alpha) * p_l

    elif ustar < 0:

        # contact moving left, we need to understand the R and *R
        # states

        # Note: transverse velocity only jumps across contact
        ut_state = ut_r

        # define eigenvalues
        lambda_r = un_r + c_r
        lambdastar_r = ustar + cstar_r

        if pstar > p_r:
            # the wave if a shock -- find the shock speed
            sigma = (lambda_r + lambdastar_r) / 2.0

            if sigma > 0.0:
                # shock is moving to the right -- solution is *R state
                rho_state = rhostar_r
                un_state = ustar
                p_state = pstar

            else:
                # solution is R state
                rho_state = rho_r
                un_state = un_r
                p_state = p_r

        else:
            # the wave is a rarefaction
            if lambda_r < 0.0 and lambdastar_r < 0.0:
                # rarefaction fan is moving to the left -- solution is
                # R state
                rho_state = rho_r
                un_state = un_r
                p_state = p_r

            elif lambda_r > 0.0 and lambdastar_r > 0.0:
                # rarefaction fan is moving to the right -- solution is
                # *R state
                rho_state = rhostar_r
                un_state = ustar
                p_state = pstar

            else:
                # rarefaction spans x/t = 0 -- interpolate
                alpha = lambda_r / (lambda_r - lambdastar_r)

                rho_state = alpha * rhostar_r + (1.0 - alpha) * rho_r
                un_state = alpha * ustar + (1.0 - alpha) * un_r
                p_state = alpha * pstar + (1.0 - alpha) * p_r

    else:  # ustar == 0

        rho_state = 0.5 * (rhostar_l + rhostar_r)
        un_state = ustar
        ut_state = 0.5 * (ut_l + ut_r)
        p_state = pstar

    # species now
    if nspec > 0:
        if ustar > 0.0:
            xn = q_l[iX:iX + nspec]
        elif ustar < 0.0:
            xn = q_r[iX:iX + nspec]
        else:
            xn = 0.5 * (q_l[iX:iX + nspec] + q_r[iX:iX + nspec])

    # are we on a solid boundary?
    if is_solid:
        un_state = 0.0

    q_int[irho] = rho_state
    if idir == 1:
        q_int[iu] = un_state
        q_int[iv] = ut_state
    else:
        q_int[iu] = ut_state
        q_int[iv] = un_state

    q_int[ip] = p_state

    if nspec > 0:
        q_int[iX:iX + nspec] = xn

    return q_int


@njit(cache=True)
def riemann_hllc(idir, ng,
                 idens, ixmom, iymom, iener, irhoX, nspec,
                 gamma, U_l, U_r, is_solid=False):
    r"""
    This is the HLLC Riemann solver.  The implementation follows
    directly out of Toro's book.  Note: this does not handle the
    transonic rarefaction.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    ng : int
        The number of ghost cells
    nspec : int
        The number of species
    idens, ixmom, iymom, iener, irhoX : int
        The indices of the density, x-momentum, y-momentum, internal energy density
        and species partial densities in the conserved state vector.
    lower_solid, upper_solid : int
        Are we at lower or upper solid boundaries?
    gamma : float
        Adiabatic index
    U_l, U_r : ndarray
        Conserved state on the left and right cell edges.

    Returns
    -------
    out : ndarray
        Conserved flux
    """

    nvar = U_l.shape[0]

    F = np.zeros(nvar)

    smallc = 1.e-10
    smallp = 1.e-10

    U_state = np.zeros(nvar)

    # primitive variable states
    rho_l = U_l[idens]

    # un = normal velocity; ut = transverse velocity
    if idir == 1:
        un_l = U_l[ixmom] / rho_l
        ut_l = U_l[iymom] / rho_l
    else:
        un_l = U_l[iymom] / rho_l
        ut_l = U_l[ixmom] / rho_l

    rhoe_l = U_l[iener] - 0.5 * rho_l * (un_l**2 + ut_l**2)

    p_l = rhoe_l * (gamma - 1.0)
    p_l = max(p_l, smallp)

    rho_r = U_r[idens]

    if idir == 1:
        un_r = U_r[ixmom] / rho_r
        ut_r = U_r[iymom] / rho_r
    else:
        un_r = U_r[iymom] / rho_r
        ut_r = U_r[ixmom] / rho_r

    rhoe_r = U_r[iener] - 0.5 * rho_r * (un_r**2 + ut_r**2)

    p_r = rhoe_r * (gamma - 1.0)
    p_r = max(p_r, smallp)

    # compute the sound speeds
    c_l = max(smallc, np.sqrt(gamma * p_l / rho_l))
    c_r = max(smallc, np.sqrt(gamma * p_r / rho_r))

    # Estimate the star quantities -- use one of three methods to
    # do this -- the primitive variable Riemann solver, the two
    # shock approximation, or the two rarefaction approximation.
    # Pick the method based on the pressure states at the
    # interface.

    p_max = max(p_l, p_r)
    p_min = min(p_l, p_r)

    Q = p_max / p_min

    rho_avg = 0.5 * (rho_l + rho_r)
    c_avg = 0.5 * (c_l + c_r)

    # primitive variable Riemann solver (Toro, 9.3)
    factor = rho_avg * c_avg
    # factor2 = rho_avg / c_avg

    pstar = 0.5 * (p_l + p_r) + 0.5 * (un_l - un_r) * factor
    ustar = 0.5 * (un_l + un_r) + 0.5 * (p_l - p_r) / factor

    # rhostar_l = rho_l + (un_l - ustar) * factor2
    # rhostar_r = rho_r + (ustar - un_r) * factor2

    if Q > 2 and (pstar < p_min or pstar > p_max):

        # use a more accurate Riemann solver for the estimate here

        if pstar < p_min:

            # 2-rarefaction Riemann solver
            z = (gamma - 1.0) / (2.0 * gamma)
            p_lr = (p_l / p_r)**z

            ustar = (p_lr * un_l / c_l + un_r / c_r +
                     2.0 * (p_lr - 1.0) / (gamma - 1.0)) / \
                     (p_lr / c_l + 1.0 / c_r)

            pstar = 0.5 * (p_l * (1.0 + (gamma - 1.0) * (un_l - ustar) /
                                  (2.0 * c_l))**(1.0 / z) +
                           p_r * (1.0 + (gamma - 1.0) * (ustar - un_r) /
                                  (2.0 * c_r))**(1.0 / z))

            # rhostar_l = rho_l * (pstar / p_l)**(1.0 / gamma)
            # rhostar_r = rho_r * (pstar / p_r)**(1.0 / gamma)

        else:

            # 2-shock Riemann solver
            A_r = 2.0 / ((gamma + 1.0) * rho_r)
            B_r = p_r * (gamma - 1.0) / (gamma + 1.0)

            A_l = 2.0 / ((gamma + 1.0) * rho_l)
            B_l = p_l * (gamma - 1.0) / (gamma + 1.0)

            # guess of the pressure
            p_guess = max(0.0, pstar)

            g_l = np.sqrt(A_l / (p_guess + B_l))
            g_r = np.sqrt(A_r / (p_guess + B_r))

            pstar = (g_l * p_l + g_r * p_r -
                     (un_r - un_l)) / (g_l + g_r)

            ustar = 0.5 * (un_l + un_r) + \
                0.5 * ((pstar - p_r) * g_r - (pstar - p_l) * g_l)

            # rhostar_l = rho_l * (pstar / p_l + (gamma - 1.0) / (gamma + 1.0)) / \
            #     ((gamma - 1.0) / (gamma + 1.0) * (pstar / p_l) + 1.0)
            #
            # rhostar_r = rho_r * (pstar / p_r + (gamma - 1.0) / (gamma + 1.0)) / \
            #     ((gamma - 1.0) / (gamma + 1.0) * (pstar / p_r) + 1.0)

    # estimate the nonlinear wave speeds

    if pstar <= p_l:
        # rarefaction
        S_l = un_l - c_l
    else:
        # shock
        S_l = un_l - c_l * np.sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) *
                                   (pstar / p_l - 1.0))

    if pstar <= p_r:
        # rarefaction
        S_r = un_r + c_r
    else:
        # shock
        S_r = un_r + c_r * np.sqrt(1.0 + ((gamma + 1.0) / (2.0 / gamma)) *
                                   (pstar / p_r - 1.0))

    #  We could just take S_c = u_star as the estimate for the
    #  contact speed, but we can actually do this more accurately
    #  by using the Rankine-Hugonoit jump conditions across each
    #  of the waves (see Toro 10.58, Batten et al. SIAM
    #  J. Sci. and Stat. Comp., 18:1553 (1997)
    S_c = (p_r - p_l + rho_l * un_l * (S_l - un_l) - rho_r * un_r * (S_r - un_r)) / \
        (rho_l * (S_l - un_l) - rho_r * (S_r - un_r))

    # figure out which region we are in and compute the state and
    # the interface fluxes using the HLLC Riemann solver
    if S_r <= 0.0:
        # R region
        U_state[:] = U_r[:]

        F[:] = consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec,
                        U_state)

    elif S_c <= 0.0 < S_r:
        # R* region
        HLLCfactor = rho_r * (S_r - un_r) / (S_r - S_c)

        U_state[idens] = HLLCfactor

        if idir == 1:
            U_state[ixmom] = HLLCfactor * S_c
            U_state[iymom] = HLLCfactor * ut_r
        else:
            U_state[ixmom] = HLLCfactor * ut_r
            U_state[iymom] = HLLCfactor * S_c

        U_state[iener] = HLLCfactor * (U_r[iener] / rho_r +
                                       (S_c - un_r) * (S_c + p_r / (rho_r * (S_r - un_r))))

        # species
        if nspec > 0:
            U_state[irhoX:irhoX + nspec] = HLLCfactor * \
                U_r[irhoX:irhoX + nspec] / rho_r

        # find the flux on the right interface
        F[:] = consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec,
                        U_r[:])

        # correct the flux
        F[:] = F[:] + S_r * (U_state[:] - U_r[:])

    elif S_l < 0.0 < S_c:
        # L* region
        HLLCfactor = rho_l * (S_l - un_l) / (S_l - S_c)

        U_state[idens] = HLLCfactor

        if idir == 1:
            U_state[ixmom] = HLLCfactor * S_c
            U_state[iymom] = HLLCfactor * ut_l
        else:
            U_state[ixmom] = HLLCfactor * ut_l
            U_state[iymom] = HLLCfactor * S_c

        U_state[iener] = HLLCfactor * (U_l[iener] / rho_l +
                                       (S_c - un_l) * (S_c + p_l / (rho_l * (S_l - un_l))))

        # species
        if nspec > 0:
            U_state[irhoX:irhoX + nspec] = HLLCfactor * \
                U_l[irhoX:irhoX + nspec] / rho_l

        # find the flux on the left interface
        F[:] = consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec,
                        U_l[:])

        # correct the flux
        F[:] = F[:] + S_l * (U_state[:] - U_l[:])

    else:
        # L region
        U_state[:] = U_l[:]

        F[:] = consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec,
                        U_state)

        # we should deal with solid boundaries somehow here

    return F


@njit(cache=True)
def consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec, U_state):
    r"""
    Calculate the conservative flux.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    gamma : float
        Adiabatic index
    idens, ixmom, iymom, iener, irhoX : int
        The indices of the density, x-momentum, y-momentum, internal energy density
        and species partial densities in the conserved state vector.
    nspec : int
        The number of species
    U_state : ndarray
        Conserved state vector.

    Returns
    -------
    out : ndarray
        Conserved flux
    """

    F = np.zeros_like(U_state)

    u = U_state[ixmom] / U_state[idens]
    v = U_state[iymom] / U_state[idens]

    p = (U_state[iener] - 0.5 * U_state[idens] * (u * u + v * v)) * (gamma - 1.0)

    if idir == 1:
        F[idens] = U_state[idens] * u
        F[ixmom] = U_state[ixmom] * u + p
        F[iymom] = U_state[iymom] * u
        F[iener] = (U_state[iener] + p) * u

        if nspec > 0:
            F[irhoX:irhoX + nspec] = U_state[irhoX:irhoX + nspec] * u

    else:
        F[idens] = U_state[idens] * v
        F[ixmom] = U_state[ixmom] * v
        F[iymom] = U_state[iymom] * v + p
        F[iener] = (U_state[iener] + p) * v

        if nspec > 0:
            F[irhoX:irhoX + nspec] = U_state[irhoX:irhoX + nspec] * v

    return F
