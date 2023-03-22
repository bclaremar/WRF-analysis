import numpy as np

def convpar(EPTU,EPTSU,zU,zp2,TU):

#calculate LEQ
  #print(EPTU)
  #print(EPTSU)
  a, =np.nonzero(EPTSU<EPTU)
  #print(a)
  #print(a.size)
  #print(len(a))
  if len(a)<1:
    LFC=np.nan
    LEQ=np.nan
    LFC2=np.nan
    LEQ2=np.nan
    TEQ=np.nan
  else:
    #print(a.size)
    a0=a[0]
    a1=a[-1]

    LFC=zU[a0]
    LEQ=zU[a1]
    TEQ=TU[a1]

   # print(LEQ)
   # print(TEQ)
    LFC2=zp2[a0]
    LEQ2=zp2[a1]

  return LFC, LEQ, TEQ, LFC2, LEQ2
