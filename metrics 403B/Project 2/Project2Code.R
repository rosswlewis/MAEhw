setwd("C:/Users/rossw/Documents/MAE Program/Q2/Applied Econometrics 403B/Project 2")

moneySupply = read.csv("M2NS.csv", header = TRUE)
cpi = read.csv("CPIAUCNS.csv", header = TRUE)

moneySupply$Date <- as.Date(moneySupply$DATE, format= "%Y-%m-%d")
moneySupply = moneySupply[,c('Date','M2NS')]
cpi$Date <- as.Date(cpi$DATE, format= "%Y-%m-%d")
cpi = cpi[,c('Date','CPIAUCNS')]

head(moneySupply)
head(cpi)

#cpi = ts(subset(cpi, Date > "1980-11-01" & Date <  "2007-12-01"))[,c('Date','CPIAUCNS')]
#moneySupply = ts(subset(moneySupply, Date > "1980-11-01" & Date <  "2007-12-01"))[,c('Date','M2NS')]



library(forecast)
library(ggplot2)
library(tseries)

#a
tsdisplay(moneySupply,main="Money Supply")
ggplot(data = moneySupply, aes(x = Date, y = M2NS)) +
  geom_line(color = "#00AFBB", size = 2)

tsdisplay(cpi,main="CPI")
ggplot(data = cpi, aes(x = Date, y = CPIAUCNS)) +
  geom_line(color = "#00AFBB", size = 2)


cpiTs = ts(subset(cpi, Date > "1980-11-01" & Date <  "2007-12-01"), frequency=12)
moneySupplyTs = ts(subset(moneySupply, Date > "1980-11-01" & Date <  "2007-12-01"), frequency=12)
data = merge(cpiTs,moneySupplyTs)
head(data)

#b
msModel=tslm(moneySupplyTs~trend+season)
plot(stl(moneySupplyTs[,'M2NS'], s.window="periodic"))
summary(msModel)

cpiModel=tslm(cpiTs~trend+season)
plot(stl(cpiTs[,'CPIAUCNS'], s.window="periodic"))
summary(cpiModel)

#c
plot(msModel$fitted.values[,'M2NS'])
plot(msModel$residuals[,'M2NS'])

plot(cpiModel$fitted.values[,'CPIAUCNS'])
plot(cpiModel$residuals[,'CPIAUCNS'])

#e
tsdisplay(msModel$residuals,main="Money Supply")
tsdisplay(cpiModel$residuals,main="CPI")


#f
library(strucchange)
plot(efp(msModel$residuals~1, type = "Rec-CUSUM"))
plot(efp(cpiModel$residuals~1, type = "Rec-CUSUM"))

#g
y=recresid(msModel$residuals~1)
plot(y, pch=16,ylab="Recursive Residuals")

y=recresid(cpiModel$residuals~1)
plot(y, ylab="Recursive Residuals")


#h



#i
plot(forecast(msModel,h=12),shadecols="oldstyle")
plot(forecast(cpiModel,h=12),shadecols="oldstyle")


#j
head(data)
library(vars)
var_model=VAR(data[,c('CPIAUCNS','M2NS')],p=4)
summary(var_model)


#k
irf(var_model)
plot(irf(var_model, n.ahead=36))


#l
head(moneySupplyTs)
gmon<-ts(moneySupplyTs[,2],start=1908,freq=12)
gcpi<-ts(cpiTs[,2],start=1908,freq=12)
grangertest(gcpi ~ gmon, order = 8)
grangertest(gmon ~ gcpi, order = 8)


#m
varPredict = predict(object=var_model, n.ahead=12)
plot(varPredict)
