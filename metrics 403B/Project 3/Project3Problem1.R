library(AER)
data("USAirlines")
head(USAirlines)
library(plyr)

typeof(USAirlines)

#1 - a
count(USAirlines,'firm')
#don't need to remove any firms

#1 - b
summary(USAirlines)

#1 - c
logout2 <- (log(USAirlines$output))^2
mod = lm(log(cost) ~ log(output) + logout2 + log(price) + load, data = USAirlines)
summary(mod)

#1 - d - 1
library(fastDummies)
full = dummy_cols(USAirlines,select_columns = c('year','firm'))
head(full)

year = dummy_cols(USAirlines,select_columns = c('year'))
mod1 = lm(log(cost) ~ . + 
           log(output) + logout2 + log(price) + load - 
           firm - year - output - cost - price - load - year_1970, data = year)
summary(mod1)

#1 - d - 2
firm = dummy_cols(USAirlines,select_columns = c('firm'))
mod2 = lm(log(cost) ~ . + 
           log(output) + logout2 + log(price) + load - 
           firm - year - output - cost - price - load - firm_1, data = firm)
summary(mod2)

#1 - d - 3
mod3 = lm(log(cost) ~ . + 
           log(output) + logout2 + log(price) + load - 
           firm - year - output - cost - price - load - year_1970 - firm_1, data = full)
summary(mod3)
#Fewer statistically significant columns when more are added, but better R-squared


#1 - e
summary(mod1)
summary(mod3)
#With both time and firm effects, the time effect coefficients are less significant and have a smaller absolute value
plot(mod1$coefficients[grepl('year',names(mod1$coefficients))],xlab='predictor', ylab='coefficient')
plot(mod3$coefficients[grepl('year',names(mod3$coefficients))],xlab='predictor', ylab='coefficient')


#1 - f
library(plm)
#rand <- plm(data = full,log(cost) ~ log(output) + logout2 + log(price) + load +
#              year_1971 + year_1972 + year_1973 + year_1974 + year_1975 + 
#              year_1976 + year_1977 + year_1978 + year_1979 + year_1980 + year_1981 + 
#              year_1982 + year_1983 + year_1984 + firm_2 + firm_3 + firm_4 + 
#              firm_5 + firm_6, model = 'random')
with <- plm(data = full,log(cost) ~ log(output) + logout2 + log(price) + load)
rand <- plm(data = full,log(cost) ~ log(output) + logout2 + log(price) + load,model='random')
summary(rand)
phtest(with, rand)

#1 - g
#no stististical significant difference (right???)