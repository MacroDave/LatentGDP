data {
  int<lower=1> T; // number of observations
  int<lower=1> J; // dimension of observations
  vector[J] Y[T]; // observations
  vector[J] Zero; // a vector of Zeros (fixed means of observations)
}
parameters {
  vector[T] Ytrue; // unobserved 'true' GDP series
  real mu; // Mean growth rate of 'latent 'true' GDP
  real<lower=0, upper=1> rho; // AR(1) Coef on 'true' GDP series
  real<lower = 0> tau; //variance of true GDP series   
  cholesky_factor_corr[J] Lcorr;  
  vector<lower=0>[J] sigma; 
}

model {
  #priors
  Ytrue[1] ~ normal(.7,1);
  tau ~ cauchy(0,5);
  mu ~ normal(0.8, 1);
  rho ~ normal(0.5, 1);
  sigma ~ cauchy(0, 5);
  Lcorr ~ lkj_corr_cholesky(4);

  #likelihood
  for(t in 2:T) {
    Ytrue[t] ~ normal(mu*(1-rho) + rho*Ytrue[t-1], tau);
    Y[t]-Ytrue[t] ~ multi_normal_cholesky(Zero, diag_pre_multiply(sigma, Lcorr));
  }
}

generated quantities {
  matrix[J,J] Omega;
  matrix[J,J] Sigma;
  Omega = multiply_lower_tri_self_transpose(Lcorr);
  Sigma = quad_form_diag(Omega, sigma); 
}
