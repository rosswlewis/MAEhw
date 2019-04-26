rm(list=ls())
setwd("C:/Users/rossw/Documents/MAE Program/Q2/Machine Learning 425/Final Project")
library('gtrendsR')
library("readxl")
data('state')
library(data.table)
library(quantmod)
library(Quandl)
library(blsAPI)
library(rjson)
library(corrplot)

#GROUP TODO
#incorporate alcohol consumption data
#finalize list of google terms

Predicting Unemployment with Google Trends Data









#code
# random forest
# linear regression
# neural net
#paper
#susan
# description of data
# description of problem
#Ross
# methods used
# results
#June
# conclusion
# figures





# categories:
#geography
#music
#social capital
#bridging/bonding social capital
#morals
# fairness ingroup harm purity authority
# fairness
# 
# i respect
#trust
#social norms, social identity

#DONE

#personality traits
# openness concienciousness extraversion agreeableness neuroticism
# good books
# online class
# events near me
# i agree
# (anxiety)
#jokes
# jokes
#sports
# football
# soccer
# baseball
# basketball
#abortion
# how to have a miscarriage
# abortion pills
# pregnency clinic
#religion
# church
# synagogue
# mosque
#voting
# where to vote
#victims
# my dad hit me
#porn
# porn
#islamaphobia
# muslim terrorist
#ideology
# liberalism socialism conservatism communism anarchism libertarianism
#sex
# sexless marriage
# I love my girlfriend's boobs
# pain during sex
#depression
# find a therapist
# counseling
# anxious
# anxiety help
#sexuality
# am i gay
# gay porn
#parental concerns
# is my son/daughter gifted
# is my son/daugher overweight
#racism
# cracker
# nigger
# beaner
# chink
#pregnancy
# my wife is pregnant
# pregnancy
#online dating
# online dating
# dating app




# google trends already normalizes for populatoin (normalized by total google searches)



#cato institute freedom in the 50 states
ef = read.csv("Freedom_In_The_50_States_2018.csv",header = T)
ef$State = ef$ï..State
keeps <- c("State", "Year",'Fiscal.Policy','Regulatory.Policy','Personal.Freedom','Economic.Freedom','Overall.Freedom')
ef = ef[keeps]
ef = ef[which(ef$Year > 2003),]
head(ef)

addUS = function(st){
  return(paste(c("US-",st),collapse=''))
}
removeUS = function(st){
  return(substring(st, 4, 5))
}






#RI Department of Labor and Training
unem = read.csv("annavgunem.csv",header = T)
colnames(unem)[colnames(unem) == 'ï..State'] <- 'State'
for(name in colnames(unem)){
  if(substring(name, 1, 1) == 'X'){
    colnames(unem)[colnames(unem) == name] <- substring(name, 2, 5)
  }
}
unem = unem[which(unem$State != 'United States'),]
unem <- subset(unem, select = -c(2000,2001,2002,2003,2017))
#drops <- c("2000","2001","2002","2003","2017")
#unem = unem[ , !(names(unem) %in% drops)]
unem = na.omit(unem)
unem = unem[which(unem$State != 'District of Columbia'),]
unem = unem[which(unem$State != 'Source: Bureau of Labor Statistics'),]
unem = unem[which(unem$State != ''),]

ef[,"unemployment"] <- NA
for(row in 1:nrow(unem)){
  for(col in c('2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2007', '2006', '2005', '2004')){
    #print(factor(unem$State[row]))
    #print(unem[row,col])
    ef[which(ef$State == as.character(factor(unem$State[row])) & ef$Year == col),]$unemployment = unem[row,col]
  }
}
head(ef)


#NIAAA alcohol consumption data
alc = read.csv("alcohol consumption.csv",header = T, strip.white=TRUE)
head(alc)
colnames(alc)[1] <- 'State'
colnames(alc)[2] <- 'Year'
alc = alc[which(alc$State != 'District of Columbia'),]
alc <- subset(alc, select = -c(2000,2001,2002,2003,2017))
ef = merge(ef,alc,by.x = c("State","Year"),by.y = c('State','Year'))


head(ef)



getTopState = function(query){
  toTest = state.abb[state.abb != 'AL']
  topState = addUS('AL')
  
  while(length(toTest) > 0){
    nextStates = c()
    while(length(toTest) > 0 & length(nextStates) < 4){
      nextStates = append(nextStates,addUS(toTest[1]))
      toTest = toTest[toTest != toTest[1]]
    }
    nextStates = append(nextStates,topState)
    bystate = gtrends(keyword=query, geo=nextStates,time = "2004-01-01 2016-12-31")$interest_over_time
    print(toTest)
    
    topState = bystate[which.max(bystate$hits),]$geo
    print(topState)
  }
  
  return(removeUS(topState))
}

getAllStates = function(query,topState){
  print(topState)
  toMine = state.abb[state.abb != topState]
  topState = addUS(topState)
  byState = c()
  while(length(toMine) > 0){
    nextStates = c()
    while(length(toMine) > 0 & length(nextStates) < 4){
      nextStates = append(nextStates,addUS(toMine[1]))
      toMine = toMine[toMine != toMine[1]]
    }
    nextStates = append(nextStates,topState)
    print(toMine)
    
    if(length(byState) > 0){
      #print(1)
      byState = rbind(byState, gtrends(keyword=query, geo=nextStates,time = "2004-01-01 2016-12-31")$interest_over_time)
    } else {
      #print(2)
      byState = gtrends(keyword=query, geo=nextStates,time = "2004-01-01 2016-12-31")$interest_over_time
    }
    #print(head(byState))
  }
  
  for(row in 1:nrow(byState)){
    byState[row,]$geo = state.name[which(state.abb == removeUS(byState$geo[row]))]
  }
  print(head(byState))
  colnames(byState)[colnames(byState) == 'year'] <- 'Year'
  colnames(byState)[colnames(byState) == 'geo'] <- 'State'
  print(head(byState))
  print('end of function')
  
  return(byState)
}
#gtrends(keyword='anxiety help', geo=c('US-WI'),time = "2004-01-01 2016-12-31")$interest_over_time
#getAllStates('anxiety help','WY')
#gtrends(keyword='anxiety help', geo=c('US-WI','US-WY'),time = "2004-01-01 2016-12-31")$interest_over_time




queries = c()
#didn't work
#,'joomla article','flash player 64',,'free tower defense game'
#
#low/no data
#'failblog','watch full episodes of','pogo scrabble','gigi spice','ifate','video tube',
#''solution center','emo girls','black celebrity','submit your',
#
#worked
#'church','anxiety',
#''64 bit','pregnancy',
#''jobs in''dating app','online dating','chink','racism','nigger','good books',
#'online class','events near me','jokes','football','religion','where to vote'


#online dating
queries = c('online dating')
for(query in queries){
  print(Sys.time())
  print(query)
  topState = getTopState(query)
  metric = getAllStates(query,topState)
  head(metric)
  DT <- data.table(date = as.IDate(metric$date), metric[,c('hits','State')])
  #temp = data.frame(DT)
  #temp = transform(temp, hits = as.integer(hits))
  #metric = setNames(aggregate(temp[, c('hits')], list(year(temp$date),temp$State), mean),c('year','State','hits'))
  metric = data.frame(DT[,list(mean=mean(as.integer(hits))),by=list(year(date),State)])
  
  ef = merge(ef,metric, by.x=c('Year','State'),by.y=c('year','State'))
  colnames(ef)[colnames(ef) == 'mean'] = gsub(" ", "", query, fixed = TRUE)
  print(head(ef))
}
ef[is.na(ef)] <- 0
write.csv(ef, file = "fullData.csv")

corrplot(cor(ef[,!names(ef) %in% c("Year", "State")]))


head(ef)
#data.frame(DT[,list(mean=mean(as.integer(hits))),by=list(year(date),State)])

tempef = ef
tempef[is.na(tempef)] <- 0

drops <- c('failblog','watch full episodes of','pogo scrabble','gigi spice','ifate','video tube',
'solution center','emo girls','black celebrity','submit your')
ef = ef[ , !(names(ef) %in% drops)]
ef$failblog

#options(datatable.optimize=1)


#Quandl.api_key("NADHW2XP9vuXetEcxGMq")
#gdp = Quandl("FRED/GDP", start_date="2004-01-01", end_date="2016-12-31", collapse="annual")
#gdp

#head(ef)
#max(ef$Year)

#state.abb
#state.name