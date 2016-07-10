library(lubridate)
completedata<-read.table("household_power_consumption.txt",sep = ";",header = TRUE,dec = ".",stringsAsFactors=FALSE,na = "?")
data <- completedata[completedata$Date %in% c("1/2/2007","2/2/2007") ,]
data$Global_active_power<-as.numeric(data$Global_active_power)
data$Date<-dmy(data$Date)
data$datetime<-strptime(paste(data$Date, data$Time),"%Y-%m-%d %H:%M:%S")
png("plot2.png",width = 480 , height = 480,units = "px")
plot(data$datetime,data$Global_active_power,ylab = "Global Active Power(kilowatts)",xlab = "",main="",type = "l")
dev.off()
