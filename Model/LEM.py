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

class LEM():

      
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
        
        response = result.loc[initial_date:final_date]
        response = self.changeformat(response)

        return response
        
#--------------------------------------------------------------------------------------
#   Metodo para cambiar keys de datestemp a string y de pandas a dict.
    def changeformat(self,response):
        dates = [str(response.index[i]) for i in range(0,len(response.index))]
        response = response.to_dict()

       
        responseq ={dates[i]: list(response['q'].values())[i] for i in range(0,len(dates))}
        responserun ={dates[i]: list(response['run'].values())[i] for i in range(0,len(dates))}
        responseprecprom ={dates[i]: list(response['precprom'].values())[i] for i in range(0,len(dates))}
        responsepetprom ={dates[i]: list(response['petprom'].values())[i] for i in range(0,len(dates))}
        responsesnow ={dates[i]: list(response['snow'].values())[i] for i in range(0,len(dates))}
        responsemelting={dates[i]: list(response['melting'].values())[i] for i in range(0,len(dates))}

        response = {'q':responseq, 'run':responserun, 'precprom':responseprecprom, 'petprom':responsepetprom, 'snow':responsesnow, 'melting':responsemelting}
        return response
#------------------------------------------------------

#------------------------------------------------------
#Modelo MELCA

    def run_melca(self,inputs):
        base_dir = os.getcwd()
        # %0- FICHEROS DE ENTRADA
        fichpar='Param_Chambo_niv7.xlsx' #%fichero de parametros
        fichdat='series_Chambo_niv7.xlsx' #%fichero de series
        fichres1='res_MELCA.xlsx' #%resultados generales
        fichres2='res_series.xlsx' #%resultados-series
        vcres=73 #%id cuenca para grÃ¡ficas

        #%1-LECTURA DE DATOS
        modelname='fun_MELCA_v1'
        #%1.1- Lee fichero de param con topologia de la red
        #matpar=readtable(fichpar)
        matpar=LEM().obtain_data(inputs)
        matpar=pd.DataFrame(matpar)
        vc=matpar.id #cÃ³digos de las cuencas del fichero de parametros
        vc_str=[str(list(vc)[i]) for i in range(0,len(vc))]
        vd=matpar.fin #codigos cuencas destino
        
        vecs0=matpar.s0
        vecfcp=matpar.fcp
        vecfce=matpar.fce
        area=matpar.area

        nc=len(vc) # %nÂºde cuencas
        ncol=max(vc); #max valor del id de cuencas (columnas de la matriz de datos de clima)


        #1.2- Lee series hidroclimÃ¡ticas:Prec, temp_min, temp_max por subcuencas
        inputs_prec = {'section':'MELCA/prec'}
        inputs_tmin = {'section':'MELCA/tmin'}
        inputs_tmax = {'section':'MELCA/tmax'}

        datosprec=LEM().obtain_data(inputs_prec)
        datosprec=pd.DataFrame.from_dict(datosprec)
        datostmin=LEM().obtain_data(inputs_tmin)
        datostmin=pd.DataFrame.from_dict(datostmin)
        datostmax=LEM().obtain_data(inputs_tmax)
        datostmax=pd.DataFrame.from_dict(datostmax)
        
        datosprec.index = pd.to_datetime(datosprec.index,unit='ms')
        datostmin.index = pd.to_datetime(datostmin.index,unit='ms')
        datostmax.index = pd.to_datetime(datostmax.index,unit='ms')

        
        vcser=datosprec.columns #códigos de las cuencas del fichero de series
        vcser_str=list(datosprec.columns)
        vcser=[int(list(vcser)[i]) for i in range(0,len(list(vcser)))]
        

        # comprueba que hay datos para todas las cuencas del fichero de param
        # if list(vc)!=list(vcser):
        #     print('Hay cuencas sin serie climática asociada')

        #Fecha comienzo de las series: 1/1/2002
        n0=datetime.date.toordinal(datetime.date(2002,1,1)) #fecha del primer dato disponible
        fechainic=datetime.date(2002,1,1)  #fecha de inicio de simulaciÃ³n
        fechafin=datetime.date(2019,12,31)  #fecha de final de simulaciÃ³n ---> la feca final estaba mal?
        ni=datetime.date.toordinal(fechainic)-n0+1
        nf=datetime.date.toordinal(fechafin)-n0+1 #-----> la posicion no coincide con la fecha final
        # fechas=datetime(fechainic):datetime(fechafin);
        nd=nf-ni+1 #n de datos


        

        #Obtiene el vector de equivalencia entre vc y vcser
        vceq=[]
        for i in range(0,nc):
            index=vcser.index(list(vc)[i])
            vceq.append(vcser[index])
        
        #Series climaticas según el orden de vc, con los factores fcp y fce
        #la primera fila y columna son cabeceros y fechas (en python no es necesario sumar 1, ya que reconoce las cabeceras)
        prec=datosprec.iloc[ni:nf][vc_str]
        factors=numpy.matlib.repmat(np.array(list(vecfcp)), nd-1,1) #tengo que restar 1 porque empieza a contar en 0
        prec=prec*factors #aplica coef. corr. de prec.
        tmin=datostmin.iloc[ni:nf][vc_str]
        tmax=datostmax.iloc[ni:nf][vc_str]

        #2- EJECUTA EL MELCA POR SUBCUENCAS
        qsims=[]

        pet=[]

        area_ac=[]
        prec_ac=[]
        pet_ac=[]
        qsim=[]
        qsim_ac=[]

        precmed=[]
        petmed=[]
        qmed=[]
        precmed_ac=[]
        petmed_ac=[]
        qmed_ac=[]
        ce_ac=[]
        smax=[]
        

        for i in range(0,len(vc)):
            #disp(['cuenca-' num2str(ic)])
            index= vc_str[i]
            datstr=pd.concat([prec[index],tmin[index],tmax[index]], axis=1)
            datstr.columns=['prec','tmin','tmax']
            
            # # par=[ tau  S0  fcp fce]
            par= np.array([matpar.tau[i],vecs0[i],vecfcp[i],vecfce[i]])
            output_melca=self.fun_MELCA_v1(par,datstr) #calcula caudales espec.
            
            qsims.append(output_melca['qsim'])
            pet.append(output_melca['pet'])
            smax.append(output_melca['smax'])
            fa=list(area)[i]*1000/(3600*24) #de mm/d a m3/s
            qsim.append(fa*output_melca['qsim'])
            precmed.append(365.25*np.mean(prec[index]))
            petmed.append(365.25*np.mean(output_melca['pet']))
            qmed.append(np.mean(fa*output_melca['qsim']))
            # %disp(['Qmed: ' num2str(qmed(i)) '  m3/s']);

        #3- CALCULA LAS SERIES ACUMULADAS
        set_vc=set(vc)
        set_vd=set(vd)

        cc= list(set_vc.difference(set_vd))

        #cc=setdiff(vc,vd) #Obtiene cuencas de cabecera (las que no son destino de ninguna)
        ncc=len(cc)
        matcon=np.identity(nc)  #matriz de conectividades acum. Fila: cuenca receptora; col:cuencas tributarias
        prec_np=np.matrix(prec.to_numpy())
        pet_np=np.transpose(np.matrix(pet))
        qsim_np=np.transpose(np.matrix(qsim))
        print(np.shape(qsim))
        print(np.shape(qsim_np))
        #Calcula matriz de conectividades
        for i in range(0,ncc):
            ipos=list(vc).index(cc[i])
            cdes=vd[ipos] #código de destino
            ipostot=[]
            ides=[list(vc).index(cdes)][0]

            while cdes>0: #codigo de la cuenca de desembocadura
                
                ides=[list(vc).index(cdes)][0]
                ipostot.append(ipos)
                matcon[ides,ipos]=1
                ipos=ides
                cdes=vd[ipos]
        
        for i in range(0,nc):
            ctrib=matcon[i,:]==1 #cuencas tributarias

            area_ac_val=sum(area[ctrib])
            prec_ac_num=np.sum(np.multiply(prec_np[:,ctrib],list(area[ctrib])),1)
            prec_ac_val=prec_ac_num/area[i]
            pet_ac_num=np.sum(np.multiply(list(area[ctrib]),pet_np[:,ctrib]),1)
            pet_ac_val=pet_ac_num/area[i]
            qsim_ac_val=np.sum(qsim_np[:,ctrib],1) 
            precmed_ac_val=365.25*np.mean(prec_ac_val,0)
            petmed_ac_val=365.25*np.mean(pet_ac_val,0)
            qmed_ac_val=np.mean(qsim_ac_val,0)
            fa=area_ac_val*1000/(3600*24)#de mm/d a m3/s
            ce_ac_val=np.divide(qmed_ac_val,(fa*precmed_ac_val/365.25))

            area_ac.append(area_ac_val)
            prec_ac.append(prec_ac_val)
            pet_ac.append(pet_ac_val)
            qsim_ac.append(np.transpose(qsim_ac_val))
            precmed_ac.append(precmed_ac_val[0,0])
            petmed_ac.append(petmed_ac_val[0,0])            
            qmed_ac.append(qmed_ac_val[0,0])
            ce_ac.append(ce_ac_val[0,0])
            


        
        q20_ac=np.quantile(qsim_ac, 0.2,2)
        q50_ac=np.quantile(qsim_ac, 0.5,2)
        q80_ac=np.quantile(qsim_ac, 0.8,2)
        q20_ac=np.transpose(q20_ac)[0]
        q50_ac=np.transpose(q50_ac)[0]
        q80_ac=np.transpose(q80_ac)[0]
        


    # %%%%%%%%%%%%%%%%%%%%%%RESULTADOS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Escribe resultados a fichero
        
        array2table = {'id':vc,'area':area,'area_ac':area_ac,'precmed':precmed,'petmed':petmed, 'qmed':qmed, 'precmed_ac':precmed_ac,
        'petmed_ac': petmed_ac,'qmed_ac': qmed_ac,'ce_ac': ce_ac,'q20_ac': q20_ac,'q50_ac': q50_ac,'q80_ac': q80_ac}       

        trestor=pd.DataFrame(array2table)
        
        # df_sim_historical_updated = pd.DataFrame(index=df_obs[initial_simulation:].index, data=np.array(res).T,
        #                                             columns=["q", "run", "precprom", "petprom",
        #                                                     "snow", "melting"])
                
        trestor.to_excel(os.path.join(base_dir, "OUTPUTS", fichres1))

        idc=vc==vcres
        print('Pmed: ', trestor['precmed_ac'][idc])
        print('PETmed: ', trestor['petmed_ac'][idc])
        print('Qmed: ', trestor['qmed_ac'][idc])
        print('Q50: ', trestor['q50_ac'][idc])
        print('Q80/Q20: ', trestor['q80_ac'][idc]/trestor['q20_ac'][idc])
        # print('CN: ', 25400/(smax(idc)+254))
            
        # %%%%%%%%%%%%%%%%%%%%%%%%%  GRÃFICOS %%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Gráficas por meses (box-whiskers)
        # tipovar 1:pre; 2:pet, 3: caudal
        self.fun_acum(prec_ac[:,idc],fechainic,fechafin,1)
        # [tspetm,tspeta]= fun_acum(pet_ac(:,idc),fechainic,fechafin,2);
        # [tsqm,tsqa]= fun_acum(qsim_ac(:,idc),fechainic,fechafin,3);

        # pm=tspm.yd;
        # qm=tsqm.yd; %Valores mensuales
        # fm=tsqm.Time;
        # qa=tsqa.yd; %Valores anuales
        # fa=tsqa.Time;
        # pa=tspa.yd;

    #-------------------------------------------------------------------------------
    #Modelo hidrológico basado en la ec. logí­stica con 4 parametros (diario)
    #*********************** READS PARS AND DATA******************************
    def fun_MELCA_v1(self,par,data):
        fce=par[3]

        prec=np.array(data['prec'])
        tmin=list(data['tmin'])
        tmax=list(data['tmax'])
        
        tmed=[(tmin[i]+tmax[i])/2 for i in range(0,len(tmin))]
        pet=[fce*12.642/365.25*(tmed[i]+17.8)*(tmax[i]-tmin[i])**0.5 for i in range(0,len(tmin))]
        # %errty=data{4};
        # %ndcal=data{5};

        qinic=statistics.mean(prec)*math.exp(-statistics.mean(pet)/statistics.mean(prec))

        kq=1/par[0] #param desfase (inverso del tiempo caracterÃ­stico)
        kp0=1/par[1] #inverso de la capac de suelo S0
        tlan=25.465*math.log(par[1])-19.494
        lan=1/tlan
        c1=1 #param de Schreiber (fijo)

        #  Initialization
        nd=len(prec)
        rsim=np.zeros((nd,1)) #runoff series
        qsim=np.zeros(nd) #discharge series
        precprom=np.zeros((nd,1))
        petprom=np.zeros((nd,1))

        #********dynamic aridity ratio*********
        precprom[0]=statistics.mean(prec)/lan
        petprom[0]=statistics.mean(pet)/lan
        rsim[0]=qinic
        qsim[0]=qinic
        aux1=math.exp(-lan)
        aux2=(1-aux1)/lan

        for i in range(1,nd):
            precprom[i]=aux1*precprom[i-1]+aux2*prec[i]
            petprom[i]=aux1*petprom[i-1]+aux2*pet[i]
        
        arin=[petprom[i]/precprom[i] for i in range(0,len(petprom))]
        #marin=harmmean(arin)
        ceq=[math.exp(-c1*arin[i]) for i in range(0,len(arin))] #Schreiber modified

        kp=kp0*prec
        req=[prec[i]*ceq[i] for i in range(0,len(prec))]
        
        for i in range(1,nd):
            #RUNOFF GENERATION
            if req[i]>0:
                rsim[i]=req[i]*math.exp(kp[i])*rsim[i-1]/(req[i]+(-1+math.exp(kp[i]))*rsim[i-1])
            else:
                rsim[i]=(ceq[i])*rsim[i-1]/((ceq[i])+kp0*rsim[i-1])
         

        for i in range(1,nd):
            # RUNOFF ROUTING
            qsim[i]=rsim[i-1]+(rsim[i]-rsim[i-1])*(1-1/kq)+math.exp(-kq)*(qsim[i-1]-rsim[i-1]+(rsim[i]-rsim[i-1])/kq)
        
        smax=np.mean(petprom) 
        
        return {'qsim':qsim,'pet':pet,'smax':smax}

        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #  self.fun_acum(prec_ac[:,idc],fechainic,fechafin,1)
    def  fun_acum(self,yd,dateinic,datefin,tvar):
        
    #  NOTA: revisar fechas y los nombres de ejes y tí­tulo del gráfico generado
        dates =[range(dateinic,datefin)]
        timetable={'date':dates,'yd':yd}
        tsyd=pd.DataFrame(timetable)
        
    # tsyd = timetable([dateinic:datefin]', yd);

    # switch tvar
    #     case {1,2} %prec/pet
    #         tspm=convert2monthly(tsyd,'Aggregation','sum');
    #         tspa=convert2annual(tsyd,'Aggregation','sum');
    #     case 3 %Caudal
    #         tspm=convert2monthly(tsyd,'Aggregation','mean');
    #         tspa=convert2annual(tsyd,'Aggregation','mean');
    # end
    # end
                    