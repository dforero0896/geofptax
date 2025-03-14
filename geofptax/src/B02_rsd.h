#ifndef B02_RSD_H_
#define B02_RSD_H_

#include <stdio.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

struct input{
 double *tr;
 double *cosm_par;
 double *pk_in;
 double sig_fog;
 gsl_interp_accel *acc_me;
 gsl_spline *spline_me;
 double *af;
 int mp;
};

void ext_bk_mp(double **tr,double **tr2, double **tr3, double **tr4, double *log_km, double *log_pkm, double *cosm_par, double redshift, int fit_full,int kp_dim, int num_tr, int num_tr2,int num_tr3,int num_tr4, double *bk_mipj_ar);
int bkeff_r(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval );
double geo_fac(double ka, double kb, double kc, double *af,double hh);
double z2_ker(double ka, double kb, double kc,double fkern, double gkern, double mua, double mub, double *cosm_par);
double z1_ker(double mu, double *cosm_par);
double g2_ker(double ka, double kb, double kc);
double* interpol_ker(double a, double f1[], double f2[], double f3[]);
double f2_ker(double ka, double kb, double kc);
double cosab(double ka, double kb, double kc);

#endif
