#ifndef GPUFIT_COMPTON_PE_CUH_INCLUDED
#define GPUFIT_COMPTON_PE_CUH_INCLUDED

#include "kn_pe.cuh"

/* Compton/Photoelectric dual-energy CT sinogram decomposition model.
*
* Decomposes dual-energy sinograms into Compton scattering (c) and
* photoelectric absorption (p) basis coefficients using the model:
*
*   sinogram(E) = -log( sum_i( spectrum(E_i) * exp(-(c*KN(E_i) + p*PE(E_i))) ) )
*
* where KN is the Klein-Nishina cross-section and PE = (60/E)^3.
*
* Parameters:
*   p[0]: Compton line integral coefficient (c)
*   p[1]: Photoelectric line integral coefficient (p)
*
* n_points must be 2 (high and low energy projections).
* point_index: 0 = high energy, 1 = low energy.
*
* user_info layout:
*   [0]  scc       : scaling factor for Compton
*   [1]  scp       : scaling factor for PE
*   [2,3]           : reserved (photon counts)
*   [4]  n_kev_h   : number of energy bins (high spectrum)
*   [5]  n_kev_l   : number of energy bins (low spectrum)
*   followed by: kev_h, kev_l, spctrm_h, spctrm_l,
*                spctrm_h_ph, spctrm_l_ph, spctrm_h_kn, spctrm_l_kn
*/

__device__ void calculate_compton_pe(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // n_fits is the total number of sinogram pixels
    // n_points is 2, corresponding to two projections, high and low per fit
    // value is the same size of n_points
    // derivative is the same size of n_parameters
    // user_info contains the two spectra and photon count

    // parameters

    REAL const * param = parameters;

    // user info
    // layout is:
    // scc, scp, pch, pcl, n_kev_h, n_kev_l, kev_h, kev_l 
    // spctrm_h, spctrm_l, spctrm_h_ph, spctrm_l_ph, spctrm_h_kn, spctrm_l_kn

    REAL * uif = (REAL *) user_info;
    REAL const scc = uif[0];
    REAL const scp = uif[1];
    // uif[2], uif[3] = photon counts (reserved, unused in forward model)
    // energy levels, here we consider 10kev to 140kev
    int const n_kev_h = (int) uif[4];
    int const n_kev_l = (int) uif[5];
    int ci = 6;
    REAL * kev_h = uif + ci; ci += n_kev_h;
    REAL * kev_l = uif + ci; ci += n_kev_l;
    // normalized spectra
    REAL * spctrm_h = uif + ci; ci += n_kev_h;
    REAL * spctrm_l = uif + ci; ci += n_kev_l;
    REAL * spctrm_h_ph = uif + ci; ci += n_kev_h;
    REAL * spctrm_l_ph = uif + ci; ci += n_kev_l;
    REAL * spctrm_h_kn = uif + ci; ci += n_kev_h;
    REAL * spctrm_l_kn = uif + ci; ci += n_kev_l;

    // value 

    REAL c = param[0], p = param[1];
    REAL h = 0, l = 0;
    REAL dh_dc = 0, dh_dp = 0;
    REAL dl_dc = 0, dl_dp = 0;
    REAL tmp = 0;
    
    c /= scc; p /= scp;

    REAL * current_derivative = derivative + point_index;

    if (point_index == 0)
    {
        // for high energy projection
        for (int i = 0; i < n_kev_h; ++i)
        {
            tmp = exp(-(c * klein_nishina(kev_h[i]) + p * photoelectric(kev_h[i])));
            h += tmp * spctrm_h[i];
            dh_dc += tmp * spctrm_h_kn[i];
            dh_dp += tmp * spctrm_h_ph[i];
        }
        value[0] = -log(h);
        current_derivative[0 * n_points] = dh_dc / h / scc;
        current_derivative[1 * n_points] = dh_dp / h / scp; 
    } 
    else
    {
        // for low energy projection
        for (int i = 0; i < n_kev_l; ++i)
        {
            tmp = exp(-(c * klein_nishina(kev_l[i]) + p * photoelectric(kev_l[i])));
            l += tmp * spctrm_l[i];        
            dl_dc += tmp * spctrm_l_kn[i];
            dl_dp += tmp * spctrm_l_ph[i];
        }
        value[1] = -log(l);
        current_derivative[0 * n_points] = dl_dc / l / scc;
        current_derivative[1 * n_points] = dl_dp / l / scp;
    }

}

#endif
