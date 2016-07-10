#install.packages("lubridate")
#install.packages("dplyr")
library(dplyr)
completedata<-read.table("household_power_consumption.txt",sep = ";",header = TRUE,dec = ".",stringsAsFactors=FALSE,na = "?")
data <- completedata[completedata$Date %in% c("1/2/2007","2/2/2007") ,]
data$Global_active_power<-as.numeric(data$Global_active_power)
png("plot1.png",width = 480 , height = 480,units = "px")
hist(data$Global_active_power,
     col = "Red" ,xlab = "Global Active Power(kilowatts)",main = "")
title(main = "Global Active Power")
dev.off()
