library(data.table)

# load data ------------------------------------------------------------------------------------------------------
# 
# as example data we use the dictionary from the Statistics Austrias classification data base
dat <- fread("https://www.statistik.at/kdb/downloads/csv/prod/ISCO08_EN_CAL.txt",
             colClasses = c("integer","character","character","character"),
             col.names = c("level","edv-code","Code08","Text"),
             encoding = "Latin-1")
dat[,level:=NULL]
dat[,`edv-code`:=NULL]

#shuffle input
dat <- dat[sample(.N)]

message("Number of unique codes: ",length(unique(dat$Code08)))

# add additional variable ------------------------------------------------------

# education (factor variable with 5 levels)
dat[,edu:=sample(1:5,.N,replace = T, prob = c(0.1,0.3,0.3,0.2,0.1))]

# citizenship (factor, 2 levels)
dat[,citizen:=sample(1:2,.N,replace = T, prob = c(0.8,0.2))]


dat[,count:=.N,by=Code08]

message("Rows input data: ", nrow(dat))
