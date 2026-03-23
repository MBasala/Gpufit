#ifndef GPUFIT_KN_PE_INCLUDED
#define GPUFIT_KN_PE_INCLUDED

__device__ REAL klein_nishina(REAL e)
{
    REAL a = e * 1.0 / 510.975;
    REAL opa = 1 + a;
    REAL op2a = 1 + 2 * a;
    REAL op3a = 1 + 3 * a;
    REAL t1 = 2 * opa / op2a;
    REAL t2 = 1 / a * log(op2a);
    REAL t3 = t2 / 2;
    REAL t4 = op3a / (op2a * op2a);
    return opa / (a * a) * (t1 - t2) + t3 - t4;
}

__device__ REAL photoelectric(REAL e)
{
    // Normalized PE basis: (60/e)^3, so PE(60) = 1.0.
    // Comparable to KN(60) ~ 1.09, giving the LM optimizer
    // balanced Jacobian columns for Compton vs PE separation.
    REAL r = 60.0 / e;
    return r * r * r;
}

#endif