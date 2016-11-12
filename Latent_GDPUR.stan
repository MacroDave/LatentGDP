data {
  int T; // number of obs activity data
  matrix[T,4] Y; //dataset of generated series
}

parameters {
  vector[T] GDPR; // unobserved 'true' GDP series
  real mu; // Mean growth rate of 'latent 'true' GDP
  real<lower = 0, upper = 1> rho; // AR(1) Coef on 'true' GDP series
  real<lower = 0> kappa; //constant in UR equation
  real<upper = 0> lambda; //loading on 'true' GDP in UR equation
  
  vector<lower = 0>[5] tau; //variance of GDP series   
  real corr_ge; //corr between true gdp, expenditure
  real corr_gi; //corr between true gdp, income
  real corr_gp; //corr between true gdp, prodn
  real<lower = 0> corr_gu; //corr between true gdp, UR
  real<lower = 0> corr_ei; //corr between expenditure, income
  real<lower = 0> corr_ep; //corr between expenditure, prodn
  real<lower = 0> corr_ip; //corr between income, production
}

transformed parameters {
  matrix[5,5] sigma;
  matrix[T,5] meanall;
  matrix[T,5] dat;

  sigma[1,1] = tau[1];
  sigma[2,1] = corr_ge;
  sigma[3,1] = corr_gi;
  sigma[4,1] = corr_gp;  
  sigma[5,1] = corr_gu;

  sigma[1,2] = sigma[2,1];
  sigma[2,2] = tau[2];
  sigma[3,2] = corr_ei;
  sigma[4,2] = corr_ep; 
  sigma[5,2] = 0;
  
  sigma[1,3] = sigma[3,1];
  sigma[2,3] = sigma[3,2];
  sigma[3,3] = tau[3];
  sigma[4,3] = corr_ip;
  sigma[5,3] = 0;
  
  sigma[1,4] = sigma[4,1];
  sigma[2,4] = sigma[4,2];
  sigma[3,4] = sigma[4,3];
  sigma[4,4] = tau[4];  
  sigma[5,4] = 0;
  
  sigma[1,5] = sigma[5,1];
  sigma[2,5] = sigma[5,2];
  sigma[3,5] = sigma[5,3];
  sigma[4,5] = sigma[5,4];   
  sigma[5,5] = tau[5];
  
  for(t in 2:T) {
    meanall[t,1] = mu*(1-rho) + rho*GDPR[t-1];
    meanall[t,2] = GDPR[t];
    meanall[t,3] = GDPR[t];
    meanall[t,4] = GDPR[t];
    meanall[t,5] = kappa + lambda*GDPR[t];
  }

  for(t in 1:T) {
    dat[t,1] = GDPR[t];
    dat[t,2] = Y[t,1];
    dat[t,3] = Y[t,2];
    dat[t,4] = Y[t,3];  
    dat[t,5] = Y[t,4];
  }
}

model {
  // priors
  GDPR[1] ~ normal(0.4,1);
  tau ~ cauchy(0,5);
  corr_ei ~ cauchy(0,2);
  corr_ep ~ cauchy(0,2);
  corr_ip ~ cauchy(0,2);
  corr_gi ~ cauchy(0,2);
  corr_gp ~ cauchy(0,2);
  corr_ge ~ cauchy(0,2);
  corr_gu ~ cauchy(0,2); 

  mu ~ normal(0.8, 1);
  rho ~ normal(0.5, 1);
  kappa ~ normal(0, 1);
  lambda ~ normal(-0.5, 1); 

  // likelihood
  for(t in 2:T) {
    target += multi_normal_lpdf(dat[t] | meanall[t], sigma);        
  }
}
