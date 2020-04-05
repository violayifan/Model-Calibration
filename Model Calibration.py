# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 03:01:42 2020

@author: viola
"""
import pandas as pd
import numpy as np
from scipy.stats import norm

from scipy import interpolate
from scipy.optimize import least_squares
from scipy.optimize import brentq

import matplotlib.pyplot as plt

#forward price
def forward_price(S,r,T):
    fp=S*np.exp(r*T)
    return fp

# interpolate discount rate    
def discount_rate(discount,period):
    f=interpolate.interp1d(discount[discount.columns[0]],discount[discount.columns[1]])
    y=f(period)
    return y
  
'''
Displaced-diffusion model Calibration
'''

# Black-Scholes model

# (1) Vanilla call/put

def BSModelCall(S,K,r,sigma,T):
    d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    Vc=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return Vc
   
def BSModelPut(S,K,r,sigma,T):
    d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    Vp=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return Vp

# calculate implied voliatility
def BSImpliedVol(func,S,K,r,T,price):
    impliedvol=brentq(lambda x:func(S, K, r, x, T)-price,-1,1)
    return impliedvol

# Black76 model

# (1) Vanilla call/put
def Black76Call(F,K,r,sigma,T):
    d1=(np.log(F/K)+0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    Vc=np.exp(-r*T)*(F*norm.cdf(d1)-K*norm.cdf(d2))
    return Vc

def Black76Put(F,K,r,sigma,T):
    d1=(np.log(F/K)+0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    Vp=np.exp(-r*T)*(K*norm.cdf(-d2)-F*norm.cdf(-d1))
    return Vp


#Bachelier model
def BachelierCall(S, K, sigma, T):
    x = (K-S)/(sigma*np.sqrt(T)*S)
    return (S-K)*norm.cdf(-x) + S*sigma*np.sqrt(T)*norm.pdf(-x)

def BachelierPut(S, K, sigma, T):
    x = (K-S)/(sigma*np.sqrt(T)*S)
    return (K-S)*norm.cdf(x) + S*sigma*np.sqrt(T)*norm.pdf(x)



# Displaced-diffusion model
# (1) Vanilla call/put
def DDCall(F,K,r,sigma,T,beta):
    Vc=Black76Call(F/beta,K+(1-beta)*F/beta,r,beta*sigma,T)
    return Vc
   
def DDPut(F,K,r,sigma,T,beta):
    Vp=Black76Put(F/beta,K+(1-beta)*F/beta,r,beta*sigma,T)
    return Vp

# implied vol when using Displaced-diffusion model to price option

def DDImpliedVol(DD_func,F,K,r,T,beta,price):
    impliedvol=brentq(lambda x:DD_func(F,K,r,x,T,beta)-price,-1,1)
    return impliedvol
    

# implied vol when using Bachelier model to price option

def BachelierImpliedVol(Bachelier_func,S,K,T,price):
    impliedvol=brentq(lambda x:Bachelier_func(S, K, x, T)-price,-1,1)
    return impliedvol




# Displaced-diffusion model Calibration
def DDcalibration(x,strikes,vols, F,T,sigma,K):
    err = 0.0
    for i, vol in enumerate(vols):
        if strikes[i]<K:
            price=DDPut(F,strikes[i],r,sigma,T,x)
            err += (vol - BSImpliedVol(BSModelPut,F,strikes[i],r,T,price) )**2
        if strikes[i]>K:
            price=DDCall(F,strikes[i],r,sigma,T,x)
            err += (vol - BSImpliedVol(BSModelCall,F,strikes[i],r,T,price) )**2   
    return err

def calDDparameter(df,F,T,sigma,K):
    initialGuess = 0.5
    res = least_squares(lambda x: DDcalibration(x,
                                              df['strike'].values,
                                              df['impliedvol'].values,
                                              F,
                                              T,
                                              sigma,K),
                    initialGuess)
    beta=res.x
    return beta

  
    


#implied vol when using SABR model to price option
def SABRImpliedVol(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom
    return sabrsigma

def sabrcalibration(x, strikes, vols, F, T,beta):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABRImpliedVol(F, strikes[i], T,
                           x[0],beta, x[1], x[2]))**2
    return err

def calsabrparameter(df,F,T,beta):
    initialGuess = [0.02, 0.2, 0.1]
    res = least_squares(lambda x: sabrcalibration(x,
                                              df['strike'].values,
                                              df['impliedvol'].values,
                                              F,
                                              T,
                                              beta),
                    initialGuess)
    alpha = res.x[0]
    rho = res.x[1]
    nu = res.x[2]
    return alpha,rho,nu


if __name__ == "__main__":
    
    #get data
    def getdata(file):
        df=pd.read_csv(file)
        return df
    
    stock=getdata("GOOG.csv")
    call=getdata("goog_call.csv")
    put=getdata("goog_put.csv")
    discount=getdata('discount.csv')
    
    
    period=(pd.to_datetime(call.expiry[0],format='%Y%m%d')-pd.to_datetime(call.date[0],format='%Y%m%d')).days
    S0=stock[stock.date==call.date[0]]["close"].values[0] 
    #r=discount_rate(discount,period)/100
    r=0
    T=period/365
    #forward price
    fp=forward_price(S0,T,r)
    K_ATM=fp

    #implied vol calculate from BS Model(using option mkt price and applying BS Model to backstep vol)
    #use mid price of bid and ask price as mkt price    
    put_liq=put[put["strike"]<K_ATM]
    put_liq["mkt_price"]=(put.best_bid+put.best_offer)/2    
    df_BS_put=pd.DataFrame([put_liq["strike"].values,put_liq[["strike","mkt_price"]].apply(lambda x:BSImpliedVol(BSModelPut,S0,x["strike"],r,T,x["mkt_price"]),axis=1)]).T
    
    call_liq=call[call["strike"]>K_ATM]
    call_liq["mkt_price"]=(call.best_bid+call.best_offer)/2
    df_BS_call=pd.DataFrame([call_liq["strike"].values,call_liq[["strike","mkt_price"]].apply(lambda x:BSImpliedVol(BSModelCall,S0,x["strike"],r,T,x["mkt_price"]),axis=1)]).T
    df_BS=df_BS_put.append(df_BS_call)
    df_BS.columns=["strike",'impliedvol']
    
    #ATM implied vol of BS Model
    BS_ATM_imp_vol=df_BS_call.iloc[0,1]
    
    #Displaced-diffusion model Calibration    
    DD_beta=calDDparameter(df_BS,S0,T,sigma=BS_ATM_imp_vol,K=K_ATM)
    #SABR Model Calibration   
    sabr_alpha,sabr_rho,sabr_nu=calsabrparameter(df_BS,S0,T,beta=0.8)
    
    
    #plot implied vol of different model
    strikes=np.linspace(300,1500,100)
    
    #calculate implied vol from BS Model(Pricing option use BS Model)
    BS_vol=[]
    sigma=BS_ATM_imp_vol
    for K in strikes[strikes<S0]:
            price=BSModelPut(S0,K,r,sigma,T)
            BS_imp_vol=BSImpliedVol(BSModelPut,S0,K,r,T,price)
            BS_vol.append([K,BS_imp_vol])

    for K in strikes[strikes>S0]:
            price=BSModelCall(S0,K,r,sigma,T)
            BS_imp_vol=BSImpliedVol(BSModelCall,S0,K,r,T,price)
            BS_vol.append([K,BS_imp_vol])
    
    
    #calculate implied vol from SABR Model(Price option use SABR Model,use the price and take into BS Model to calculate vol)
    #fix beta=0.8
    sabr_beta=0.8
    SABR_vol=[]
    for K in strikes:
        SABR_imp_vol=SABRImpliedVol(S0, K, T, sabr_alpha, sabr_beta, sabr_rho, sabr_nu)
        SABR_vol.append([K, SABR_imp_vol])
        
    #calculate implied vol from Displaced-diffusion model(calculate option price using DD model and DDsigam(use ATM price from BS Model and take it into DD Model to calculate DDsigma),
    #use the price and take into BS Model to calculate vol)
    DD_vol=[]
    DD_imvol_put=DDImpliedVol(DDPut,S0,K_ATM,r,T,DD_beta,price=BSModelPut(S0,K_ATM,r,sigma,T))
    DD_imvol_call=DDImpliedVol(DDCall,S0,K_ATM,r,T,DD_beta,price=BSModelCall(S0,K_ATM,r,sigma,T))
    
    for K in strikes[strikes<S0]:
        dd_put_price= DDPut(S0,K,r,DD_imvol_put,T,DD_beta)
        dd_im_vol=BSImpliedVol(BSModelPut,S0,K,r,T,dd_put_price)
        DD_vol.append([K,dd_im_vol])
                    
    for K in strikes[strikes>S0]:
        dd_call_price= DDCall(S0,K,r,DD_imvol_call,T,DD_beta)
        dd_im_vol=BSImpliedVol(BSModelCall,S0,K,r,T,dd_call_price)
        DD_vol.append([K,dd_im_vol])

    #calculate implied vol from  Bachelier model
    Bach_vol=[]
    Bach_imvol_put=BachelierImpliedVol(BachelierPut,S0,K_ATM,T,price=BSModelPut(S0,K_ATM,r,sigma,T))
    Bach_imvol_call=BachelierImpliedVol(BachelierCall,S0,K_ATM,T,price=BSModelCall(S0,K_ATM,r,sigma,T))

    
    for K in strikes[strikes<S0]:
        B_put_price=BachelierPut(S0, K,Bach_imvol_put, T)
        B_im_vol=BSImpliedVol(BSModelPut,S0,K,r,T,B_put_price)
        Bach_vol.append([K,B_im_vol])
                    
    for K in strikes[strikes>S0]:
        B_call_price=BachelierCall(S0, K, Bach_imvol_call, T)
        B_im_vol=BSImpliedVol(BSModelCall,S0,K,r,T,B_call_price)
        Bach_vol.append([K,B_im_vol])
    
    
    
    #plot implied vol
    fig, ax = plt.subplots()
    plt.plot(df_BS["strike"],df_BS["impliedvol"],".")
    l=["Black-Scholes","SABR","Displaced-diffusion","Bachelier model"]
    c=["b","c","r","k"]
    for i,model in enumerate([BS_vol,SABR_vol,DD_vol,Bach_vol]):
            vol_model=pd.DataFrame(model,columns=df_BS.columns)
            plt.plot(vol_model["strike"],vol_model["impliedvol"],c[i],label=l[i])
  
    plt.legend()
    plt.savefig("implied_vol from different model")
    
    fig, ax = plt.subplots()
    l=["Black-Scholes","Bachelier model"]
    c=["b","c"]
    for i,model in enumerate([BS_vol,Bach_vol]):
        vol_model=pd.DataFrame(model,columns=df_BS.columns)
        plt.plot(vol_model["strike"],vol_model["impliedvol"],c[i],label=l[i])
    
    plt.legend()
    plt.xlabel("strike")
    plt.ylabel("implied volatility")


    


    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    