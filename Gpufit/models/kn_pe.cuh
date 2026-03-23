#ifndef GPUFIT_KN_PE_INCLUDED
#define GPUFIT_KN_PE_INCLUDED

__device__ REAL klein_nishina(REAL e)
{
    // Guard: energy must be positive (keV). Clamp to 1 keV minimum
    // to prevent division-by-zero in the KN formula.
    if (e < REAL(1.0)) e = REAL(1.0);

    REAL a = e / REAL(510.975);
    REAL opa = REAL(1.0) + a;
    REAL op2a = REAL(1.0) + REAL(2.0) * a;
    REAL op3a = REAL(1.0) + REAL(3.0) * a;
    REAL t1 = REAL(2.0) * opa / op2a;
    REAL t2 = REAL(1.0) / a * log(op2a);
    REAL t3 = t2 / REAL(2.0);
    REAL t4 = op3a / (op2a * op2a);
    return opa / (a * a) * (t1 - t2) + t3 - t4;
}

__device__ REAL photoelectric(REAL e)
{
    // Normalized PE basis: (60/e)^3, so PE(60) = 1.0.
    // Comparable to KN(60) ~ 1.09, giving the LM optimizer
    // balanced Jacobian columns for Compton vs PE separation.
    if (e < REAL(1.0)) e = REAL(1.0);
    REAL r = REAL(60.0) / e;
    return r * r * r;
}

#endif
