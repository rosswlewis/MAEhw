data = read.csv('403datasetremoved.csv')
#data = read.csv('newdata.csv')

head(data)
data = within(data, rm('X'))
colnames(data)[1] <- 'date'
data = data[which(data$taylor != '#N/A'),]
data$date = as.Date(data$date, format = "%Y-%m-%d")

library(forecast)
tsData = ts(data = data$Actual.Fed.Funds, start = c(1960,1), frequency = 4)
fitARIMA <- auto.arima(tsData, trace=TRUE)
library(lmtest)
coeftest(fitARIMA)

head(data)
plot(data$smoothtaylor)
lines(data$Actual.Fed.Funds)

library(ggplot2)
ggplot(data=data) + 
  #geom_line(aes(x=date, y=smoothtaylor,col = "Taylor"), size = .5) +
  geom_line(aes(x=date, y=taylor,col = "Taylor"), size = .5) +
  geom_line(aes(x=date,y = Actual.Fed.Funds,col='Actual'), size = .5) +
  geom_line(aes(x=date,y = fitted(fitARIMA),col='ARIMA model'), size = .5) +
  labs(title = "Interest Rates Over Time",color="Interest") +
  xlab('Date') +
  ylab('Interest Rate')

futurVal <- forecast(fitARIMA,h=10, level=c(99.5))
plot(futurVal)


fitARIMA

checkresiduals(fitARIMA)
accuracy(fitARIMA)

#library(tsoutliers)
library(tseries)
jarque.bera.test(residuals(fitARIMA))
jarque.bera.test(c[1,2])

#tsData

components.ts = decompose(tsData)
plot(components.ts)
acf(tsData,lag.max=34) 
pacf(tsData,lag.max=34) 



length(tsData)
library(foreach)
windowSize = 50
numberWindows = length(tsData) - windowSize

rm(forecasts)
forecasts = foreach(i=1:numberWindows, .combine = rbind) %do%{
  y_in = tsData[1:(windowSize + i)]
  fit = auto.arima(y_in)
  f1 = forecast(fit,h = 1)
  f1 = as.numeric(f1$mean)
  f2 = tsData[(windowSize+i+1):(windowSize+i+1)]
  #print(f2)
  #print(windowSize+i+1)
  return(c(f1,f2))
}

library(Metrics)

forecasts = na.omit(forecasts)
#is.na()
#final[complete.cases(final), ]
rmse = rmse(forecasts[,1],forecasts[,2])
mape = mape(forecasts[,1],forecasts[,2])
mape
rmse

head(forecasts)

tsData[(41):(41)]
head(tsData)
f = forecast(fit,h=1)
f$mean

library(vars)
library(lmtest)
library(dplyr)
inflationgap=abs(data$InflationPCE-2)
data$inflationgap=inflationgap
head(data)
bVAR=select(data, Actual.Fed.Funds, inflationgap, gdpgap)
notaylorVAR=VAR(bVAR, lag.max = 10)
summary(notaylorVAR)
impulse=irf(notaylorVAR)
plot(impulse)
