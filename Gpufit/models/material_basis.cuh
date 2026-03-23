#ifndef GPUFIT_MATERIAL_BASIS_CUH_INCLUDED
#define GPUFIT_MATERIAL_BASIS_CUH_INCLUDED

/* Description of the calculate_material_basis function
* =====================================================
*
* Two-material basis decomposition for dual-energy CT sinograms.
*
* Unlike COMPTON_PE which uses analytic Klein-Nishina and photoelectric
* basis functions, MATERIAL_BASIS uses tabulated LAC curves for two
* known materials (e.g., water + iodine, or bone + soft tissue).
*
* The forward model for each energy spectrum is:
*   sinogram(E) = -log( sum_i( spectrum(E_i) * exp(-(a1*m1(E_i) + a2*m2(E_i))) ) )
*
* Parameters:
*   p[0] = a1: Material 1 line integral coefficient
*   p[1] = a2: Material 2 line integral coefficient
*
* user_info layout:
*   [0]  scc       : scaling factor for material 1
*   [1]  scp       : scaling factor for material 2
*   [2]  pch       : high-energy photon count (for normalization)
*   [3]  pcl       : low-energy photon count (for normalization)
*   [4]  n_kev_h   : number of energy bins for high spectrum
*   [5]  n_kev_l   : number of energy bins for low spectrum
*   followed by:
*     m1_h[n_kev_h]      : material 1 LAC at high-energy keV points
*     m1_l[n_kev_l]      : material 1 LAC at low-energy keV points
*     m2_h[n_kev_h]      : material 2 LAC at high-energy keV points
*     m2_l[n_kev_l]      : material 2 LAC at low-energy keV points
*     kev_h[n_kev_h]     : photon energies (high) — reserved
*     kev_l[n_kev_l]     : photon energies (low) — reserved
*     spctrm_h[n_kev_h]  : incident spectrum (high)
*     spctrm_l[n_kev_l]  : incident spectrum (low)
*     spctrm_h_m1[n_kev_h]: m1-weighted spectrum (high) for derivatives
*     spctrm_l_m1[n_kev_l]: m1-weighted spectrum (low) for derivatives
*     spctrm_h_m2[n_kev_h]: m2-weighted spectrum (high) for derivatives
*     spctrm_l_m2[n_kev_l]: m2-weighted spectrum (low) for derivatives
*/

__device__ void calculate_material_basis(
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
    // parameters
    REAL const * param = parameters;

    // user info
    REAL * uif = (REAL *) user_info;
    REAL const scc = uif[0];
    REAL const scp = uif[1];
    REAL const pch = uif[2];
    REAL const pcl = uif[3];
    int const n_kev_h = (int) uif[4];
    int const n_kev_l = (int) uif[5];
    int ci = 6;

    // Material curves — separate arrays for each energy grid
    REAL * m1_h = uif + ci; ci += n_kev_h;
    REAL * m1_l = uif + ci; ci += n_kev_l;
    REAL * m2_h = uif + ci; ci += n_kev_h;
    REAL * m2_l = uif + ci; ci += n_kev_l;

    // energy levels (reserved, advance index)
    ci += n_kev_h;  // kev_h
    ci += n_kev_l;  // kev_l

    // normalized spectra
    REAL * spctrm_h = uif + ci; ci += n_kev_h;
    REAL * spctrm_l = uif + ci; ci += n_kev_l;

    // pre-multiplied spectra for derivatives
    REAL * spctrm_h_m1 = uif + ci; ci += n_kev_h;
    REAL * spctrm_l_m1 = uif + ci; ci += n_kev_l;
    REAL * spctrm_h_m2 = uif + ci; ci += n_kev_h;
    REAL * spctrm_l_m2 = uif + ci; ci += n_kev_l;

    // parameters
    REAL a1 = param[0], a2 = param[1];
    REAL h = 0, l = 0;
    REAL dh_da1 = 0, dh_da2 = 0;
    REAL dl_da1 = 0, dl_da2 = 0;
    REAL tmp = 0;

    a1 /= scc; a2 /= scp;

    REAL * current_derivative = derivative + point_index;

    if (point_index == 0)
    {
        // high energy projection: -log(I / I0) where I0 = pch
        for (int i = 0; i < n_kev_h; ++i)
        {
            tmp = exp(-(a1 * m1_h[i] + a2 * m2_h[i]));
            h += tmp * spctrm_h[i];
            dh_da1 += tmp * spctrm_h_m1[i];
            dh_da2 += tmp * spctrm_h_m2[i];
        }
        value[point_index] = -log(h / pch);
        current_derivative[0 * n_points] = dh_da1 / h / scc;
        current_derivative[1 * n_points] = dh_da2 / h / scp;
    }
    else if (point_index == 1)
    {
        // low energy projection: -log(I / I0) where I0 = pcl
        for (int i = 0; i < n_kev_l; ++i)
        {
            tmp = exp(-(a1 * m1_l[i] + a2 * m2_l[i]));
            l += tmp * spctrm_l[i];
            dl_da1 += tmp * spctrm_l_m1[i];
            dl_da2 += tmp * spctrm_l_m2[i];
        }
        value[point_index] = -log(l / pcl);
        current_derivative[0 * n_points] = dl_da1 / l / scc;
        current_derivative[1 * n_points] = dl_da2 / l / scp;
    }
}

#endif
