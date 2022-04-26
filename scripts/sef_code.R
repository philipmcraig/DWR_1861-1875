library(dataresqc)
library(readr)

docdir = 'C:/Users/qx911590/Documents/'
qcdir = paste(docdir,'qc_SEF',sep='')
datadir = paste(docdir,'Data_for_SEF/',sep='')
sefdir = paste(docdir,'SEF_1861-1875',sep='')
station = "Portrush"

obs <- data.frame
obs = read_csv(paste(datadir,'Portrush_rain.csv',sep='')) # read file in Windows system

z = dim(obs) # dimensions of obs data frame, rows x columns

Data <- data.frame
Data = matrix(data=c(obs$year,obs$month,obs$day,obs$hour,obs$minute,obs$rain),
              ncol=z[2]-1,nrow=z[1])
# input data will go in here

cod_in = paste('DWRUK',toupper(station),sep='_')
sou_in = 'DWR_UKMO'

write_sef(Data,sefdir,variable="rr",cod=cod_in,nam="Portrush",
              lat="55.21",lon="-6.65",alt="",sou=sou_in,units="mm",stat="sum",period="p1day",time_offset=0,
          meta=obs$meta,metaHead="")

qc('C:/Users/qx911590/Documents/SEF_1861-1875/DWR_UKMO_DWRUK_PORTRUSH_18620211-18630530_rr.tsv',outpath=qcdir)

write_flags('C:/Users/qx911590/Documents/SEF_1861-1875/DWR_UKMO_DWRUK_SHIELDS_18620211-18750331_rr.tsv',
            paste(qcdir,'/qc_DWRUK_SHIELDS_rr_subdaily.txt',sep=''),outpath=sefdir,note='qc')
