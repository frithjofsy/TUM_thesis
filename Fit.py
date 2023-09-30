
# -*- coding: utf-8 -*-
"""
Created on Feb - 2023

@author: Narmin Ghaffari Laleh, modified by Frithjof Sy
"""

###############################################################################

import numpy as np
from scipy.optimize import curve_fit
import warnings
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import Utils as utils
import FitFunctions as ff
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Normalizer
import matplotlib as mpl
from matplotlib.lines import Line2D

###############################################################################

#<<<<<<< HEAD
# The Name of studies which are save in one folder
studies = ['Study1', 'Study2', 'Study3', 'Study4', 'Study5']
#=======
#studies = ['a', 'a', 'c', 'd', 'e']
#>>>>>>> c7994f5b057179a0511e27ccbd9dd9f379a44404
#studies = ['1', '2', '3', '4', '5']
functions = ['DoubleExponential', 'Lottka-Volterra', 'ModKuznetsov', 'GameTheory']
splits = [True, True, False, True, True] # für Studie 1, 4 und 5 braucht man es nicht unbedingt, dürfte also auch false sein.
#splits = [False, True, False, False, False]
#trends = ['Up', 'Down', 'Fluctuate']
trends = ['Resistance']
#noPars = [3, 3, 3, 4, 3, 4]
noPars = [4, 6, 3, 8]

###############################################################################

# FIND MAXIMUM OF THE DATA SETS To be able to Normaliz the Whole Tumor Dimensions
# Diesen Schritt kann man vielleicht verkleinern

maxList = []
minList = []

for studyName in studies:
    #rawDataPath = "C:\\Masterarbeit\\Narmins Code (Python)\\ImmunotherapyModels-main\\RawData\\a_m.xlsx"
    rawDataPath = os.path.join(r"C:\\Masterarbeit Sicherung 10.01.2023\\Masterarbeit\\FrithjofS Code\\RawData", studyName + '_m.xlsx')
    sind = studies.index(studyName)
    sp = splits[sind]
    #data, arms = utils.Read_Excel(rawDataPath, ArmName='Study_Arm', split=sp)
    data, arms = utils.Read_Excel(rawDataPath, ArmName = 'TRT01A', split = sp)
    filtered_Data = data.loc[data['TRLINKID'] == 'INV-T001']
    filtered_Data = filtered_Data.loc[filtered_Data['TRTESTCD'] == 'LDIAM']
    temp = list(filtered_Data['TRORRES'])
    temp = utils.Remove_String_From_Numeric_Vector(temp, valueToReplace = 0)
    maxList.append(max(temp))
    minList.append(min(temp))    
               
###############################################################################

# Fit Funtions to the Data Points
    
#maxi = np.max([288, 0])
maxi = np.max(maxList)

#<<<<<<< HEAD
#=======
#studies = ['a', 'a', 'c', 'd', 'e']
studies = ['Study1', 'Study2', 'Study3', 'Study4', 'Study5']
switch = 3
if switch == 1:
    functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
    noPars = [3, 3, 3, 4, 3, 4]
elif switch == 2:
    functions = ['DoubleExponential', 'Lottka-Volterra', 'ModKuznetsov', 'GameTheory']
    noPars = [3, 6, 7, 3]
elif switch == 3:
    functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz', 'DoubleExponential', 'Lottka-Volterra', 'ModKuznetsov','GameTheory']
    noPars = [3, 3, 3, 4, 3, 4, 3, 6, 7,3]
elif switch == 4:
    functions = ['DoubleExponential']
    noPars = [3]
elif switch == 5:
    functions = ['DoubleExponential', 'Lottka-Volterra', 'ModKuznetsov']
    noPars = [3, 6, 7]
elif switch == 6:
    functions = ['GameTheory']
    noPars = [3]
elif switch == 7:
    functions = ['GeneralBertalanffy']
    noPars = [4]
elif switch == 8:
    functions = ['ModKuznetsov']
    noPars = [7] #bei 6 halten wir einen fest, alpha
elif switch == 9:
    functions = ['GeneralGompertz']
    noPars = [4] 
elif switch == 10:
    functions = ['Gompertz']
    noPars = [3] 
elif switch == 11:
    functions = ['Lottka-Volterra']
    noPars = [6] 



#functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
#functions = ['DoubleExponential', 'Lottka-Volterra', 'ModKuznetsov', 'GameTheory']
splits = [True, True, False, True, True]
#noPars = [3, 3, 3, 4, 3, 4]
#noPars = [4, 6, 3, 8]

#>>>>>>> c7994f5b057179a0511e27ccbd9dd9f379a44404
for studyName in studies:
    continue
    countfig = 1
    sind = studies.index(studyName)    
    sp = splits[sind]    
    studyName = studies[sind]
    warnings.filterwarnings("ignore")
    normalizeDimension = True
    
    rawDataPath = os.path.join(r"C:\\Masterarbeit Sicherung 10.01.2023\\Masterarbeit\\FrithjofS Code\\RawData", studyName + '_m.xlsx')
    data, arms = utils.Read_Excel(rawDataPath, ArmName = 'TRT01A', split = sp)
    for functionToFit in functions:
        find = functions.index(functionToFit)
        noParameters = noPars[find]
        result_dict = utils.Create_Result_dict(arms, ['Resistance'], categories = ['patientID', 'rmse', 'rSquare', 'time', 'dimension', 'prediction', 'aic', 'params', 'cancer'])
        #result_dict = utils.Create_Result_dict(arms, ['Up', 'Down', 'Fluctuate'], categories = ['patientID', 'rmse', 'rSquare', 'time', 'dimension', 'prediction', 'aic', 'params', 'cancer'])
        print(functionToFit)
        print(studyName)
        
        for arm in arms:
            print(arm)        
            data_temp = data.loc[data['receivedTreatment'] == arm]    
            patientID = list(data_temp['USUBJID'].unique())
            
            for key in patientID:
                
                filteredData = data.loc[data['USUBJID'] == key]
                temp = filteredData['TRLINKID'].unique()
                temp = [i for i in temp if not str(i) == 'nan']
                temp = [i for i in temp if not '-NT' in str(i)]
        
                if  'INV-T001' in temp :
                    tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == 'INV-T001']
                    tumorFiltered_Data.dropna(subset = ['TRDY'], inplace = True)            
                    tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']
                    
                    # Limit the Data Points for 6 and bigger!
                    keysList = []
                    if len(tumorFiltered_Data) >= 6:                        
                        dimension = list(tumorFiltered_Data['TRORRES'])
                        time = list(tumorFiltered_Data['TRDY'])
                        
                        time = utils.Correct_Time_Vector(time, convertToWeek = True)
        
                        # If the value of Dimension is nan or any other string value, we replace it with zero    
                        dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace = 0)
                        
                        dimension = [x for _,x in sorted(zip(time,dimension))]
                        dimension_copy = dimension.copy()
                        if normalizeDimension:
                            dimension_copy = dimension_copy/maxi
                            #dimension_copy = dimension_copy/np.max(dimension_copy)
                            
                        trend = utils.Detect_Trend_Of_Data(dimension_copy)
                        #print(trend)
                        if trend == "Other":
                            continue
                    
                        
                        dimension = [i * i * i * 0.52 for i in dimension]
                        if normalizeDimension:
                            dimension = dimension/np.max([maxi * maxi * maxi * 0.52, 0])
                        time.sort()                        
                        cn =   list(tumorFiltered_Data['TULOC']) [0]                          
                        #param_bounds=([0] *noParameters ,[np.inf] * noParameters) # hier sind die param_bounds von 0 - inf
                        if find == 4: #originally ==4
                            param_bounds=([[0, -np.inf, 0],[np.inf, np.inf, np.inf]]) # warum hab ich hier diese param_bounds?
                        elif find == 5: #originally ==5
                            param_bounds=([[0, -np.inf, 2/3, 0],[np.inf, np.inf, 1, np.inf]])
                        elif find == 3: # originally ==3
                            param_bounds=([[0, 0, 2/3, 0],[np.inf, np.inf, 1, np.inf]])

                        
                        elif find == 0:
                            #param_bounds=([[0, 0, 0, 0, 0, 0],[100, 100, np.inf, 100, 100, np.inf]]) # last for Lotka-Volterra
                            #param_bounds=([[0, 0.01, 0],[0.01, 1, 1]])
                            param_bounds=([[0, 0.01, 0],[0.01, 1, 1]])

                        print(param_bounds)
                         #   param_bounds=([[10,1,0.01,0.01,0.01,0.01,0.01],[100,20,5,20,30,1,np.inf]]) # param_bounds for DoubleExponential
                         #   param_bounds=([[10,0.01,0.01,0.01],[100,20,30,np.inf]])
                           # param_bounds=([[10,1,0.01,0.01,0.01,0.01],[100,20,5,20,30,np.inf]]) # 5 free params
                            #param_bounds=([[10,0.01,0.01,0.01,0.01],[100,5,20,30,np.inf]]) # sigma, rho, mu, delta
                           # param_bounds=([[10,0.01,0.01,0.01],[100,20,30,np.inf]]) # sigma, mu, delta: three free params
                            #param_bounds=([[0.01,0.01,0.01,0.01],[100,100,100,100]]) # mu, delta, alpha
                          #  sigma, rho, nu, mu, delta, alpha

                        #firstDim = dimension[0:-3]
                        #firstTime = time[0:-3]
                        #time_without3 = time[:-3] # choose the amount of data points to fit to [:-3] means leave out three, etc. [:-2] means leave out 2
                        #dimension_without3 = dimension[:-3]                        
                        try:
                            fitfunc = ff.Select_Fucntion(functionToFit)
                            
                            geneticParameters = ff.generate_Initial_Parameters_genetic(fitfunc,k = noParameters, boundry = [0, 1], t = time, d = dimension)
                            fittedParameters, pcov = curve_fit(fitfunc, time, dimension, geneticParameters, maxfev = 200000, bounds = param_bounds, method = 'trf') 
                            modelPredictions = fitfunc(time, *fittedParameters)
                           # geneticParameters = ff.generate_Initial_Parameters_genetic(fitfunc,k = noParameters, boundry = [0, 1], t = time_without3, d = dimension_without3) # Hier: Parameter werden ohne die letzten drei Messwerte geschätzt
                           # fittedParameters_3, pcov = curve_fit(fitfunc, time_without3, dimension_without3, geneticParameters, maxfev = 200000, bounds = param_bounds, method = 'trf') # Hier: Parameter werden ohne die letzten drei Messwerte geschätzt
                           # modelPredictions_3 = fitfunc(time, *fittedParameters_3)    # Hier: werden wieder alle Zeitpunkte berücksichtigt.                    
                        except:
                            result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'], values = [key, time, dimension, np.nan, np.nan, np.nan, np.nan, np.nan, cn])
                            continue
                        
                        if len(set(dimension)) == 1:
                            modelPredictions = dimension
                            #modelPredictions_3 = dimension
                        else:
                            modelPredictions = fitfunc(time, *fittedParameters) 
                            #modelPredictions_3 = fitfunc(time, *fittedParameters_3)
                        
                        absError = modelPredictions - dimension
                        SE = np.square(absError)
                        temp_sum = np.sum(SE)
                        MSE = np.mean(SE)   

                        result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'], 
                                                                  values = [key, time, dimension, modelPredictions, mean_squared_error(dimension, modelPredictions),
                                                                            r2_score(dimension, modelPredictions), (2 * noParameters) - (2 * np.log(temp_sum)), fittedParameters, cn])


                        #absError = modelPredictions_3 - dimension
                        #SE = np.square(absError)
                        #temp_sum = np.sum(SE)
                        #MSE = np.mean(SE)   

                        #result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'], 
                        #                                          values = [key, time, dimension, modelPredictions_3, mean_squared_error(dimension, modelPredictions_3),
                        #                                                    r2_score(dimension, modelPredictions_3), (2 * noParameters) - (2 * np.log(temp_sum)), fittedParameters, cn])


                        plt.figure(figsize=(8,6), dpi=300)
                        plt.plot(time, modelPredictions, label='fit')
                        #plt.plot(time, modelPredictions_3, label='prediction')
                        plt.plot(time, dimension, marker='o', linestyle= ' ', label='Measurement')
                        plt.xlabel('Time (weeks)')
                        plt.ylabel('Tumor Size')
                        plt.title(os.path.join(r"R²= "+ str(r2_score(dimension, modelPredictions))))
                        #plt.title(os.path.join(r"R² for Prediction= "+ str(round(r2_score(dimension, modelPredictions_3),3)))) # r2 für die 3 weggelassenen Messpunkte.
                        plt.legend()
                        plt.savefig(os.path.join(r"C:\\Masterarbeit Sicherung 10.01.2023\\Masterarbeit\\FrithjofS Code\\Results\\" +str(functionToFit), "Figure_" + str(countfig) + str(studyName) + str(arm) + str(functionToFit) + str(key) +'.png'))
                        #plt.savefig(os.path.join(r"C:\\Masterarbeit Sicherung 10.01.2023\\Masterarbeit\\FrithjofS Code\\Results\\" +str(functionToFit), str(key) +'.png'))
                        #plt.show()
                        plt.close()
                        countfig +=1

        a_file = open(os.path.join(r"C:\Masterarbeit Sicherung 10.01.2023\Masterarbeit\FrithjofS Code\Results", functionToFit, studyName + '.pkl'), "wb")
        pickle.dump(result_dict, a_file)
        a_file.close()

        
           
###############################################################################
# Plot HeatMaps
###############################################################################

result = pd.DataFrame()
for f in functions:   
    print(f) 
    temp = []
    indices = []
    for s in studies:   
        #result_dict = pickle.load( open( r"C:\Masterarbeit Sicherung 10.01.2023\Masterarbeit\FrithjofS Code\Results" +'\\'+ f + '\\' + s + ".pkl", "rb" ) )
        result_dict = pickle.load( open( r"C:\Masterarbeit Sicherung 10.01.2023\Masterarbeit\FrithjofS Code\Results" +'\\'+ f + ' - only resistance' + '\\' + s + ".pkl", "rb" ) )
        arms = list(result_dict.keys())        
        arms.sort()
        for arm in arms:            
            for trend in trends:           
                
                if f == 'ClassicBertalanffy' and 6647907695500911616 in result_dict[arm][trend]['patientID']:
                    index = result_dict[arm][trend]['patientID'].index(6647907695500911616)
                    result_dict[arm][trend]['rSquare'][index] = 0.001
                    result_dict[arm][trend]['rmse'][index] = 1
                    print(index)
                    print('TRUE')
                elif f == 'ClassicBertalanffy' and -1094950970223831040 in result_dict[arm][trend]['patientID']:
                    index = result_dict[arm][trend]['patientID'].index(-1094950970223831040)
                    result_dict[arm][trend]['rSquare'][index] = 0.001
                    result_dict[arm][trend]['rmse'][index] = 1
                elif f == 'ClassicBertalanffy' and -8060287321299098624 in result_dict[arm][trend]['patientID']:
                    index = result_dict[arm][trend]['patientID'].index(-8060287321299098624)
                    result_dict[arm][trend]['rSquare'][index] = 0.001
                    result_dict[arm][trend]['rmse'][index] = 1
                elif f == 'ClassicBertalanffy' and -3778177877716539904 in result_dict[arm][trend]['patientID']:
                    index = result_dict[arm][trend]['patientID'].index(-3778177877716539904)
                    result_dict[arm][trend]['rSquare'][index] = 0.001
                    result_dict[arm][trend]['rmse'][index] = 1

             
                indices.append(arm + '_' + trend)
                temp.append(np.around(np.nanmean(result_dict[arm][trend]['rSquare']), 3)) # für AIC und R^2
                #print(np.sqrt(np.nanmean(result_dict[arm][trend]['rmse'])), 3) # für RMSE
                #temp.append(np.sqrt(np.nanmean(result_dict[arm][trend]['rmse']))) # für RMSE
                #temp.append(np.nanmean(result_dict[arm][trend]['aic'])) # für AIC, vorsicht mit der sqrt()!
                #temp.append(np.nanmean(result_dict[arm][trend]['rmse'])) # das dürfte dann nur der MSE sein, weil die Wurzel oben nicht berechnet wird
    result[f] = temp
    
result.index = indices
result.dropna(inplace = True)
minValuesObj = result.min(axis=1)

#tab_n = result.div(result.max(axis=1), axis=0)
tab_n = result.div(result.max(axis=0), axis=1)
cmap = sns.cm.rocket
#cmap = sns.cm.magma
mpl.rcParams['font.size'] = 20
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
#plt.figure()
plt.figure(figsize=(40, 32))
plt.tight_layout()
t = tab_n.T

ax = sns.heatmap(tab_n, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True ,
                  square = True, annot=True) # annot=True macht die Zahlen noch rein.
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xticklabels(labels = functions, rotation = 30, fontsize = 20)
plt.title('R² values for each arm', fontsize = 30, fontweight='bold' )
#plt.title('AIC values for each arm', fontsize = 40, fontweight='bold')
#plt.title('RMSE values for each arm', fontsize = 40, fontweight='bold')

###############################################################################

# Fit Example Per Study

#<<<<<<< HEAD
#=======
#studies = ['a', 'a', 'c', 'd', 'e']
#functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']

f = functions[3]
#>>>>>>> c7994f5b057179a0511e27ccbd9dd9f379a44404
arm = [0,0,1,0,0,1,0,1,2,0,1,1,0,0,0]
item =[3,3,0,3,8,1,3,1,0,3,0,4,0,2,1]
f = functions[3]
i = 0

for s in studies: 
    continue
    #result_dict = pickle.load( open( r"Path To the results\\" + f + '\\' + s + ".pkl", "rb" ) )
    result_dict_full = pickle.load( open( r"C:\Masterarbeit Sicherung 10.01.2023\Masterarbeit\FrithjofS Code\Results" +'\\'+ f + ' - only resistance' + '\\' + s + ".pkl", "rb" )  )
    arms = list(result_dict_full.keys())   
    arms.sort()
    
    for t in trends:
        a = arm[i]
        it = item[i]
        if t == 'Up':
            c = '#d73027'
        elif t == 'Down':
            c = '#1a9850'
        else:
            c = '#313695'
            
        #utils.Plot(result_dict, result_dict_full,  arms[a], t, it, isPrediction = True, dotCol = c, i = i)
        utils.Plot(result_dict_full, arms[a], t, it, isPrediction = True, dotCol = c, i = i)
        i = i+1
        
###############################################################################

# Calculate MAE for final point prediction
        
result = pd.DataFrame()

for f in functions:
    temp = []
    indices = []
    for s in studies:   
        #result_dict = pickle.load( open( r"Path To the results\\" + f + '\\' + s + ".pkl", "rb" ) )
        result_dict = pickle.load( open( r"C:\Masterarbeit Sicherung 10.01.2023\Masterarbeit\FrithjofS Code\Results" +'\\'+ f + ' - only resistance' + '\\' + s + ".pkl", "rb" ) )
        #result_dict = pickle.load( open( r"C:\Masterarbeit Sicherung 10.01.2023\Masterarbeit\FrithjofS Code\Results" +'\\'+ f + ' - leave out 3 - pkl updated' + '\\' + s + ".pkl", "rb" ) )

        arms = list(result_dict.keys())        
        arms.sort()
        for arm in arms:
            for trend in trends:
                indices.append(arm + '_' + trend)
                content = result_dict[arm][trend]['dimension']
                g = []
                for i in range(len(content)):
                    if not str(result_dict[arm][trend]['prediction'][i]) == 'nan':
                        g.append(abs(content[i][-1] - result_dict[arm][trend]['prediction'][i][-1]))
                temp.append(np.nanmean(g))                
    result[f] = temp
    
result.index = indices
result.dropna(inplace = True)
minValuesObj = result.min(axis=1)
tab_n = result.div(result.max(axis=1), axis=0)
cmap = sns.cm.rocket
mpl.rcParams['font.size'] = 30
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
#plt.figure(figsize=(20,16))
plt.figure()
plt.tight_layout()
t = tab_n.T
ax = sns.heatmap(tab_n, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True ,
                  square = True, annot = True)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xticklabels(labels = functions, rotation = 30,fontsize = 30 )
plt.title('MAE values for each arm', fontsize = 40, fontweight = 'bold')
#plt.title('MAE values (leave out 3)', fontsize = 40, fontweight = 'bold')
              
###############################################################################
#<<<<<<< HEAD
# Plot the observed values versus predicted values
###############################################################################

colors = ['#313695', '#3288bd', '#f46d43', '#66c2a5', '#5e4fa2', '#9e0142']
fig = plt.figure()
ax = fig.add_subplot(111)
s = studies[3]

for item in range(10):
    first = True
    ind = 0
    plt.figure()
    trend = trends[0]
    for f in functions: 
       # result_dict = pickle.load( open( r"Path To the results\\" + f + '\\' + s + ".pkl", "rb" ))
        result_dict = pickle.load( open( r"C:\Masterarbeit Sicherung 10.01.2023\Masterarbeit\FrithjofS Code\Results" +'\\'+ f + ' - only resistance' + '\\' + s + ".pkl", "rb" ) )

        arms = list(result_dict.keys())
        arm = arms[1]
        if first:
            plt.plot(result_dict[arm][trend]['time'][item], result_dict[arm][trend]['dimension'][item], 'ro', markersize = 15)
            first = False
        print(result_dict[arm][trend]['prediction'][item])    
        plt.plot(result_dict[arm][trend]['time'][item], result_dict[arm][trend]['prediction'][item], color = colors[ind],
                 markeredgewidth = 3, linewidth = 5)    
        ind += 1
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30) 
    
# Create Legend
    
fig = plt.figure()
ax = fig.add_subplot(111)      
custom_lines = [Line2D([0], [0], color = colors[0], lw=7),
               Line2D([0], [0], color = colors[1], lw=7),
               Line2D([0], [0], color = colors[2], lw=7),
               Line2D([0], [0], color = colors[3], lw=7),
               Line2D([0], [0], color = colors[4], lw=7),
               Line2D([0], [0], color = colors[5], lw=7)]

ax.legend(custom_lines, functions,  fontsize = 30)

ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.draw()

    
###############################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#=======


#>>>>>>> c7994f5b057179a0511e27ccbd9dd9f379a44404
