import requests
import json
import os
import numpy as np
import pandas as pd
import datetime
import socket
from urllib3.connection import HTTPConnection


HTTPConnection.default_socket_options = (
    HTTPConnection.default_socket_options + [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.SOL_TCP, socket.TCP_KEEPIDLE, 45),
        (socket.SOL_TCP, socket.TCP_KEEPINTVL, 10),
        (socket.SOL_TCP, socket.TCP_KEEPCNT, 6)
    ]
)

class LEM:

      
    #Método para obtener los datos de los diferentes métodos que hay en la API 
    def obtain_data(self,inputs,param=None):  
      
        url = "http://127.0.0.1:5000/api/{0}".format(inputs["section"])

        headers = {'Accept':  'application/json'}
        s = requests.Session()
        r1 = s.get(url, headers=headers) 
        data1=json.loads(r1.text)
        if param:
            s = requests.Session()
            r1 = s.get(url, headers=headers,  json =param) 
            data1=json.loads(r1.text)
        return data1
    
# #-----------------------------------------------------------------------------
# #     MODELO LEM
# #-----------------------------------------------------------------------------
    def fun_LEM_pred(self,par, data, condinic):

        fcp=par[4]   # precipitation correction factor # ----------------------->?
        km=par[5]/24 #degree-day parameter (melting process)
        tc1=par[6]   # threshold parameter (for snow onset)
        tc2=par[7]   # threshold parameter (for snow melting)

        qobs=data[0]
        prec=fcp*data[1]
        pet=data[2]
        temp=data[3]

        # Initialization
        nd=len(qobs);
        snow=np.zeros(nd);    #snow tank series
        qm=np.zeros(nd);      #melting discharge series
        precltot=np.zeros(nd);
        snow[0]=condinic[4]; #initial snow depth
        qm[0]=condinic[5];   #initial melting discharge

        #Obtains liquid and solid precipitation (precl and precs)
        mp=1;                                     #smoothing constant for snow generation law (ºC)
        precl = prec/(1+np.exp(-(temp-tc1)/mp));  #Liquid                                            # ----------------------->?
        precs=prec-precl                          #Solid precipitation

        #*************************SNOW TANK************************
        mm=1                                      #smoothing constant for the melting law (ºC)
        tm=(temp-tc2)/mm;
        for i in range(1,nd):
            qm[i]=min(snow[i-1], km*mm*(tm[i]+np.log(1+np.exp(-tm[i])))) #melting.
            #Assumption: temp are averaged values over the timestep (not instantaneous)
            snow[i]=snow[i-1]+(precs[i]-qm[i]);     #snow storage
            precltot[i]=precl[i]+qm[i];             #liquid precip + melting (feeds the standard LEM)


        #***********************Calls standard LEM**********************
        datstr=[qobs,precltot,pet]
        qsim,rsim,precprom,petprom=self.fun_LEM1(par[0:5],datstr,condinic)

        # ndcal=0;
        # err=fun_errq(qobs(ndcal+1:nd),qsim(ndcal+1:nd),1);

        return qsim, rsim, precprom, petprom, snow, qm

    def fun_LEM1(self,par,data,condinic):
    # Time series: datstr={qobs,prec,etp,temp}
    # Parameters par=[a  tau	p0	lan	fcp	ks	tc1	tc2]
    # Condiciones iniciales condinic=[q0, r0, precprom0,petprom0,st0,qm0]

        c1=par[0]
        kq=1/par[1]; #param routing-inverse of tau (1/h)
        kp=1/par[2] #growth rate-inverse of P0 (1/mm)
        lan=1/(par[3]*24);#param largo plazo (dias)
        qobs=data[0]
        prec=data[1]
        pet=data[2]

        # Initialization
        nd=len(qobs)
        rsim=np.zeros(nd)
        qsim=np.zeros(nd)
        precprom=np.zeros(nd)
        petprom=np.zeros(nd)

        #********EXPONENTIAL SMOOTHING to obtain the dynamic aridity ratio*********
        qsim[0]=condinic[0]; #initial discharge
        rsim[0]=condinic[1]; #initial runoff
        precprom[0]=condinic[2]; #initial precprom
        petprom[0]=condinic[3]; #initial petprom

        aux1=np.exp(-lan);
        aux2=(1-aux1)/lan;
        for i in range(1,nd):
            precprom[i]=aux1*precprom[i-1]+aux2*prec[i]
            petprom[i]=aux1*petprom[i-1]+aux2*pet[i]

        arin=petprom/precprom
        #************************LOGISTIC EQUATION*********************************
        ceq=np.exp(-c1*arin);
        req=ceq*prec;
        kpt=np.exp(kp*prec);
        


        for i in range(1, nd):
            #RUNOFF GENERATION
            if req[i]>1e-6:
                rsim[i]=rsim[i-1]*req[i]*kpt[i]/(req[i]+(kpt[i]-1)*rsim[i-1])
            else:
                rsim[i]=ceq[i]*rsim[i-1]/(ceq[i]+kp*rsim[i-1])
            
            

            # RUNOFF ROUTING
            qsim[i]=rsim[i-1]+(rsim[i]-rsim[i-1])*(1-1/kq)+np.exp(-kq)*(qsim[i-1]-rsim[i-1]+(rsim[i]-rsim[i-1])/kq)
        #qsim(i)=((2-kq)*qsim(i-1)+kq*(rsim(i)+rsim(i-1)))/(2+kq);

        return qsim,rsim,precprom,petprom

#--------------------------------------------------------------------------------------

    def LEM_simulation(self,inputs,param):
        
        base_dir = os.getcwd()
        data = self.obtain_data(inputs)
        station_name = param['Nombre']
        station_id = 'C0B{}'.format(param['id'])
        data_pd=pd.DataFrame.from_dict(data, orient='index')
        df_parameters=data_pd.T

        date_time_str_initial =inputs['initial date']
        date_time_str_final =inputs['final date']

        final_date = datetime.datetime.strptime(date_time_str_final, '%y-%m-%d %H:%M:%S')
        initial_date = datetime.datetime.strptime(date_time_str_initial, '%y-%m-%d %H:%M:%S')

        
        parameters = df_parameters[df_parameters.Nombre == station_name].values[0]  
          
        # Parámetros del modelo
        parameters = parameters[3:11] # par_tau 	par_p0 	par_lan 	par_fcp 	par_ks 	par_tc1 	par_tc2 	Qthr_1 	Qthr_2
        
        #Reordeno a mano porque la API me cambia el orden...
        parameters[0]=df_parameters[df_parameters.Nombre == station_name].values[0][6] #par_a
        parameters[1]=df_parameters[df_parameters.Nombre == station_name].values[0][11]#par_tau
        parameters[2]=df_parameters[df_parameters.Nombre == station_name].values[0][10]#par_p0
        parameters[3]=df_parameters[df_parameters.Nombre == station_name].values[0][9]#par_lan
        parameters[4]=df_parameters[df_parameters.Nombre == station_name].values[0][7]#par_fcp
        parameters[5]=df_parameters[df_parameters.Nombre == station_name].values[0][8]#par_ks
        parameters[6]=df_parameters[df_parameters.Nombre == station_name].values[0][12]#par_tc1
        parameters[7]=df_parameters[df_parameters.Nombre == station_name].values[0][13]#par_tc2
        #parameters[8]=df_parameters[df_parameters.Nombre == station_name].values[0][2]#Qthr_1
        #parameters[9]=df_parameters[df_parameters.Nombre == station_name].values[0][3]#Qthr_2
        #parameters[10]=df_parameters[df_parameters.Nombre == station_name].values[0][4]#Qthr_3
        print(parameters)
        #--------------------------------------------------------------------------
        # Cargar datos de simulaciones historicas
        #--------------------------------------------------------------------------
        inputs1 = {'section':'LEM/historical'}
    
        df_sim_historical =pd.DataFrame.from_dict(self.obtain_data(inputs1,param))
        
        df_sim_historical.index = pd.to_datetime(df_sim_historical.index,unit='ms')
        

        # Estados iniciales = última línea de las simulaciones históricas
        initial_states = df_sim_historical.iloc[-1].values # último valor de q 	run 	precprom 	petprom 	snow 	melting
        initial_date_sim = df_sim_historical.index[-1]  #última fecha
        
        print(initial_date_sim)
        
        #--------------------------------------------------------------------------
        # Cargar datos de observaciones
        #--------------------------------------------------------------------------
            
        inputs2 = {'section':'LEM/observado'}
        df_obs =pd.DataFrame.from_dict(self.obtain_data(inputs2,param))
        df_obs.index = pd.to_datetime(df_obs.index,unit='ms')
            
            

        #--------------------------------------------------------------------------
        # Actualizar las simulaciones historicas
        #--------------------------------------------------------------------------
            
        # Actualización de los estados: simulamos entre el útlimo dato de estado disponible y el último valor observado    
        assert (initial_date_sim <=  df_obs.index[-1])
            
            
        # Inicio simulación: 1 hora después del último dato observado
        initial_simulation = initial_date_sim + pd.DateOffset(hours=1)
        print(initial_simulation)    
            
        forcings = [df_obs[initial_simulation:].Caudal.values,
                    df_obs[initial_simulation:].Prec.values,
                    df_obs[initial_simulation:].ETP.values, # ----------------------->?
                    df_obs[initial_simulation:].Temp.values]

        #si la fecha de fin es mayor a la de las simulaciones ya realizadas, entonces tengo que ejecutar LEM para calcular las nuevas simulaciones
        if final_date>initial_date_sim:
    
                
            
            res = self.fun_LEM_pred(par=parameters, 
                            data=forcings, 
                            condinic=initial_states)
            
            # Orden:
            #     qsim, rsim, precprom, petprom, snow, qm

            # Guardamos los resultados: en producción probablemente estos datos
            # se irán añadiendo a una base de datos
            

            df_sim_historical_updated = pd.DataFrame(index=df_obs[initial_simulation:].index, data=np.array(res).T,
                                                    columns=["q", "run", "precprom", "petprom",
                                                            "snow", "melting"])
                
            df_sim_historical_updated.to_excel(os.path.join(base_dir, "OUTPUTS", "{}_{}_historical_updated.xlsx".format(station_id, station_name)))

        
        else:
            df_sim_historical_updated = None
        
        if final_date> df_obs.index[-1]:
            
            print('LEM in forecast mode')

            #--------------------------------------------------------------------------
            # Cargar datos de predicciones
            #--------------------------------------------------------------------------
            inputs3 = {'section':'LEM/forecast'}
            df_forcing_forecast =pd.DataFrame.from_dict(self.obtain_data(inputs3,param))
            df_forcing_forecast.index = pd.to_datetime(df_forcing_forecast.index,unit='ms')
            #--------------------------------------------------------------------------
            # Simulación en modalidad de predicciones
            #--------------------------------------------------------------------------
            initial_states = df_sim_historical_updated.iloc[-1].values
            initial_date_forecast = df_sim_historical_updated.index[-1] 
            initial_simulation_forecast = initial_date_forecast + pd.DateOffset(hours=1)
            
            print(initial_date_forecast)    
            print(initial_simulation_forecast)    
            
            forcings = [df_forcing_forecast[initial_date_forecast:].Caudal.values,
                        df_forcing_forecast[initial_date_forecast:].Prec.values,
                        df_forcing_forecast[initial_date_forecast:].ETP.values,
                        df_forcing_forecast[initial_date_forecast:].Temp.values]
            
            res = self.fun_LEM_pred(par=parameters, 
                        data=forcings, 
                        condinic=initial_states)
            df_forecast = pd.DataFrame(index=df_forcing_forecast[initial_date_forecast:].index,
                                    data=np.array(res).T, 
                                    columns=["q", "run", "precprom", "petprom",
                                                    "snow", "melting"])
            
            df_forecast.to_excel(os.path.join(base_dir, "OUTPUTS", "{}_{}_hydrological_forecast.xlsx".format(station_id, station_name)))
        else:
            df_forecast = None

        #--------------------------------------------------------------------------
        # Informe de predicciones
        #--------------------------------------------------------------------------
        #Rango total de fechas: 2020-10-01 00:00:00 -- 2021-12-03 11:00:00 

        frames = [df_sim_historical, df_sim_historical_updated, df_forecast]
        result = pd.concat(frames)

        return result.loc[initial_date:final_date]
        


