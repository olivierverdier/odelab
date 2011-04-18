Exponential Solvers
===================

.. math::
    φ_{\ell}(z) =  \int_0^1 \ee^{z (1-x)} \frac{x^{\ell-1}}{(\ell - 1)!}  \dd x

.. math::

	φ_{\ell}(z) = \frac{1}{\ell!} + z φ_{\ell+1}(z)

.. math::

	φ_1(z) = \frac{\ee^z-1}{z}

By induction:

.. math::

	φ_{\ell} = ∑_{k=0}^{∞} \frac{z^k}{(\ell+k)!}

Indeed:

.. math::

	∑_{k=0}^{∞} \frac{z^k}{(\ell+k)!}  = \frac{1}{\ell!} + z φ_{\ell+1}(z)

.. math::

	\iff ∑_{k=1}^{∞} \frac{z^k}{(\ell+k)!} =  z φ_{\ell+1}(z)
So:

.. math::

	z\left(φ_{\ell+1}(z) - ∑_{k=0}^{∞} \frac{z^k}{(\ell+1+k)!}\right) = 0
and clearly:

.. math::

	φ_{\ell+1}(0) = 1

.. math::

	d^{\ell} := \frac{d!}{(2d-\ell)!}


.. math::

	D_j^\ell :=  d^{\ell} (-1)^j   \frac{ (2d + \ell -j)!}{j! (d-j)!}

.. .. math::

.. A_{j+1}^\ell = \frac{-(d-j)}{(2d+\ell -j)(j+1)} A_{j}^{\ell}

So one may start with:

.. math::

	D_0^0 = 1

and compute for :math:`0≤j<d`:

.. math::

	D_{j+1}^0 = \frac{-(d-j)}{(2d -j)(j+1)} D_{j}^{0}

then, for :math:`0≤ \ell < \ell_{\max}`:

.. math::

	D_{j}^{\ell+1} = (2d-\ell)(2d+\ell+1-j) D_{j}^{\ell}

.. \[C^{\ell}_j := \frac{1}{(\ell + j)!}\]

.. math::

	C_j := \frac{1}{j!}

..  Similarly:
..  .. math::

..	C^{0}_0 = 1
..  .. math::

..	C^{0}_{j+1} = \frac{1}{j+1}C^0_j
..  .. math::

..	C^{\ell+1}_j = \frac{1}{\ell+1}C^{\ell}_j

..  .. math::

..	D^{\ell} =  A^{\ell}

.. math::

	N^{\ell} =  C^{\ell} \star D^{\ell} 

.. automodule:: odelab.phi_pade
	:members:
