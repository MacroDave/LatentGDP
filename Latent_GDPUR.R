library(ggthemes); library(stringr);library(reshape2);library(rstan); library(dplyr); library(ggplot2); library(Quandl); library(zoo);
options(mc.cores = parallel::detectCores())

#real gdp millions chained dollars - qtly seasonally adj
GDPA <- Quandl("AUSBS/5206002_EXPENDITURE_VOLUME_MEASURES_A2304402X") %>% 
  mutate(Date = as.Date(Date)) %>% 
  arrange(Date) %>% 
  mutate(GDPA = 100*log(Value/lag(Value, 1))) %>% 
  dplyr::select(Date, GDPA)

#real gdp  by expenditure, millions chained dollars - qtly seasonally adj
GDPE <- Quandl("AUSBS/5206024_SELECTED_ANALYTICAL_SERIES_A2302586T") %>% 
  mutate(Date = as.Date(Date)) %>% 
  arrange(Date) %>% 
  mutate(GDPE = 100*log(Value/lag(Value, 1))) %>% 
  dplyr::select(Date, GDPE)

#real gdp  by income, millions chained dollars - qtly seasonally adj
GDPI <- Quandl("AUSBS/5206024_SELECTED_ANALYTICAL_SERIES_A2302587V") %>% 
  mutate(Date = as.Date(Date)) %>% 
  arrange(Date) %>% 
  mutate(GDPI = 100*log(Value/lag(Value, 1))) %>% 
  dplyr::select(Date, GDPI)

#real gdp  by production, millions chained dollars - qtly seasonally adj
GDPP <- Quandl("AUSBS/5206024_SELECTED_ANALYTICAL_SERIES_A2302588W") %>% 
  mutate(Date = as.Date(Date)) %>% 
  arrange(Date) %>% 
  mutate(GDPP = 100*log(Value/lag(Value, 1))) %>% 
  dplyr::select(Date, GDPP)

#unemployment rate - monthly seasonally adj
UNR <- Quandl("AUSBS/6202001_A84423050A") %>%
  mutate(Date = as.Date(Date)) %>% 
  arrange(Date) %>% 
  mutate(UNR = (Value+lag(Value,1)+lag(Value,2))/3) %>%
  mutate(UNR = UNR-lag(UNR,3)) %>%
  dplyr::select(Date, UNR)

# Join series together, and set missing values to -999 (Stan doesn't like NAs)
full_data <- list(GDPA, GDPE, GDPI, GDPP, UNR) %>%
  Reduce(function(dtf1,dtf2) left_join(dtf1,dtf2,by="Date"), .)

#Pick Sample
full_data <- subset(full_data, Date>"1979-12-31")

# Standardize columns (without -999s) and replace -999s
full_data_tmp <- full_data[,-(1:2)]
full_data_tmp[is.na(full_data_tmp)] <- -999

# Now run the model
latent_model <- stan(file = "Latent_GDPUR.stan", 
                  data = list(T = nrow(full_data),
                              Y = full_data_tmp), control = list(adapt_delta = 0.99), chains = 4)

summarised_state <- as.data.frame(latent_model) %>% 
  select(contains("GDPR")) %>%
  melt() %>% 
  group_by(variable) %>% 
  summarise(median = median(value),
            lower = quantile(value, 0.025),
            upper = quantile(value, 0.975)) %>% 
  mutate(GDPA = full_data[,2],
       Dates = full_data$Date)

summarised_state %>% 
  ggplot(aes(x = Dates)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "orange", alpha = 0.3) +
  geom_line(aes(y = median), colour="red") +
  geom_line(aes(y = GDPA)) +
  annotate("text", x = as.Date("2004-01-01"), y = 2.5, label = "GDP growth (GDPA)") +
  annotate("text", x = as.Date("2004-01-01"), y = 2.2, label = "Underlying GDP growth", colour = "red") +
  ggthemes::theme_economist() +
  ggtitle("Latent GDP Growth Model")

# Print estimated parameters from the model
print(latent_model, pars = c("mu", "rho", "kappa", "lambda", "sigma"))

# End ---------------------------------------------------------------------
#shinystan::launch_shinystan(latent_model)