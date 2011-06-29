Exponential Solvers
===================

Shifted Exponentials
**********************

The *shifted exponentials*, usually called *φ-functions*, are defined by the following series:

.. math::
	φ_{\ell}(z) = ∑_{k=0}^{∞} \frac{z^k}{(\ell+k)!} \qquad \ell \in\NN

Notice in particular that

.. math::
    φ_0(z) = \ee^z

We observe the following recursion relation

.. math::
   φ_{\ell}(z) = \frac{1}{\ell!} + z ∑_{k=1}^{∞}\frac{x^{k-1}}{(\ell+k)!} = \frac{1}{\ell!} + z φ_{\ell+1} 

This allows to prove the useful identity, valid for :math:`\ell≥1`:

.. math::
    φ_{\ell}(z) =  \int_0^1 \ee^{z (1-x)} \frac{x^{\ell-1}}{(\ell - 1)!}  \dd x

Indeed, by integration by parts, the recursion relation is the same as above, and for :math:`\ell=1` one has

.. math::
	φ_1(z) = \frac{\ee^z-1}{z}



Padé Approximations
*******************

One computes Padé approximations of the form

.. math::
    φ_{\ell}(z) = \frac{N(z)}{D(z)}

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
