import glob
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
#import io
#import shutil 

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy import signal
import scipy.stats as ss

from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from scipy import signal
from scipy.integrate import simpson

from antropy import app_entropy
from nolds import dfa

class lacevApp:
    def __init__(self,): 
        global  GfilePathR, GfilePathF, GfilePathG, GfilePathFe

    def startLacevApp(self,houseRoot):
        global  GfilePathR, GfilePathF, GfilePathG, GfilePathFe
        
        
        self.filePathR =  os.path.join(houseRoot, 'raw') 
        self.filePathF =  os.path.join(houseRoot, 'filtered') 
        self.filePathM =  os.path.join(houseRoot, 'mlResults') 
        self.filePathG =  os.path.join(houseRoot, 'graphics') 
        self.filePathFe =  os.path.join(houseRoot, 'features') 
        try:
            self.filepathC = [file for file in os.listdir(self.filePathR) if os.path.isdir(os.path.join(self.filePathR, str(file))) ]
            GfilePathR = self.filePathR 
            GfilePathF = self.filePathF
            GfilePathG = self.filePathG
            GfilePathFe = self.filePathFe
            msg={'erro':True}
            return msg 
        except FileNotFoundError as er:   
            msg={'erro':False, 'msg':er}
            return msg 

    def startDir(self):
        msg=[]
        if len(self.filepathC) == 0 or isinstance(self.filepathC, list) == False:
                msg.append('raw directory incorrectly created. \n')
                
                os._exit(0)
        else:
            msg.append('Creating directories... \n') 

        if len(self.filepathC) == 0 or isinstance(self.filepathC, list) == False:
                msg.append('raw directory incorrectly created. \n')
                os._exit(0)
        else:
            msg.append('Creating directories... \n' )  
            

        try:
            os.mkdir(self.filePathF)
            msg.append("Creating Directory 'filtered'... \n")
        except FileExistsError:
            msg.append("Directory 'filtered' already exists... \n")
        try:
            os.mkdir(self.filePathM)
            msg.append("Creating Directory 'mlResults'... \n")
        except FileExistsError:
            msg.append("Directory 'mlResults' already exists... \n")
        try:
            os.mkdir(self.filePathG)
            msg.append("Creating Directory 'graphics'... \n")
        except FileExistsError:
            msg.append("Directory 'graphics' already exists... \n")
        try:
            os.mkdir(self.filePathFe)
            msg.append("Creating Directory 'features'... \n")
        except FileExistsError:
            msg.append("Directory 'features' already exists... \n")

        for dicte in self.filepathC:
            if os.path.isdir(os.path.join(self.filePathR, str(dicte)) ): 
                dirPath =  os.path.join(self.filePathF, str(dicte)) 
                try:
                    os.mkdir(dirPath)
                    msg.append("Creating Directory '{dicte}' ... \n")
                except FileExistsError:
                    msg.append(f"Directory '{dicte}' already exists... \n")
        return msg

    def startgraph(self):
        msg=[]
        if len(self.filepathC) == 0 or isinstance(self.filepathC, list) == False:
                msg = 'raw directory incorrectly created. \n'
                msg={'erro':False, 'msg':msg}
                return msg
        

        if len(self.filepathC) == 0 or isinstance(self.filepathC, list) == False:
                msg = 'raw directory incorrectly created. \n'
                msg={'erro':False, 'msg':msg}
                return msg        
      
        try:
            os.mkdir(self.filePathF)
            msg = "Data not found... \n"
            msg={'erro':False, 'msg':msg}
            return msg
        except FileExistsError:
            msg = "Directory 'filtered' already exists... \n"
        try:
            os.mkdir(self.filePathM)
            msg = "Data not found... \n"
            msg={'erro':False, 'msg':msg}
            return msg
        except FileExistsError:
            msg = "Directory 'mlResults' already exists... \n"
        try:
            os.mkdir(self.filePathG)
            msg = "Data not found... \n"
            msg={'erro':False, 'msg':msg}
            return msg
        except FileExistsError:
            msg = "Directory 'graphics' already exists... \n"
        try:
            os.mkdir(self.filePathFe)
            msg = "Data not found... \n"
            msg={'erro':False, 'msg':msg}
            return msg
        except FileExistsError:
            msg = "Directory 'features' already exists... \n"

        for dicte in self.filepathC:
                dirPath =   os.path.join(self.filePathF, str(dicte))
                try:
                    os.mkdir(dirPath)
                    msg = "Data not found... \n"
                    msg={'erro':False, 'msg':msg}
                    return msg
                except FileExistsError:
                    msg = f"Directory '{dicte}' already exists... \n"
        msg={'erro':True}
        return msg

    def explore(self, path):
        FILEBROWSER_PATH = os.path.join(os.getenv('WINDIR'), 'explorer.exe')
        path = os.path.normpath(path)

        if os.path.isdir(path):
            subprocess.run([FILEBROWSER_PATH, path])
        

    def getfilepathC(self):
        return self.filepathC
    def getfilepatR(self):
        return self.filePathR
    def getfilePathG(self):
        return self.filePathG
    def getGfilePathFe(self):
        return self.filePathFe

class Features:
  def __init__(self): 
    
    global  GfilePathR, GfilePathF, GfilePathFe
    
    self.pathRaw = GfilePathR
    self.pathFiltered = GfilePathF   
    self.pathFeatures = GfilePathFe
      
  def startFeature(self, dataP, minCut, timeStart, classe, nameClass, SampleRate, results,  text_box, fileN, progBar, stage=False)-> np.array:
    """
    Parameters
    ----------
    dataP : ndarray
             The array of data to be filtered.
    minCut : int
             How many times will the series be cut in minutes
    timeStart: int
               If you have a specific time that you want to specify. If it is 0 the time count does not count for hours.
    classe: int
            class identifier for classification
    nameClass: str
               class name
    SampleRate: int
                Capture frequency
    results: array
             Array that contains the features
    stage: str
           This is an electroma-specific variable. If false can be ignored
    """
    text_box.insert(1.0, 'Calculating features for '+fileN+'. Please, wait...\n')
    text_box.pack()    
    totalData = len(dataP) # Number of points collected
    minu = SampleRate*60*minCut # Minutes in amount of points
    numCut = len(dataP)//minu # How many times will the series be cut
    cminu = timeStart

    if(totalData >= minu): # To cut the series, the number of points must be greater than the amount of cut
        
      Ncut = minu
      third = int(Ncut//3) # The number of points will be interspersed with one third of the previous series to preserve the information 
      
      for i in range(1,numCut+1):             
          if i == 1: # if it's 1 round, the series starts at 0 and final at the amount to be cut
              starts = 0
              final = Ncut
          else: # from round 2 onwards, the start will be where the previous round ended minus the intersection, the final will be the start plus the cut number
              starts = final - third
              final = starts + Ncut
             
          dataQ = dataP[starts:final]  

          vzero = abs(dataQ.mean()) 
          
          if (len(dataQ) > 100) and (float(vzero) > 0.0): # the cut series must have more than 100 points and not have the average value zeroed
            
            progBar['value'] += numCut//100
            if progBar['value'] > 100: progBar['value'] = 0
            progBar.update_idletasks()

            numpyArray = np.array(dataQ).ravel() 
            
            apenC = app_entropy(numpyArray, order=2) 
            dfaC = dfa(numpyArray, fit_exp='poly', overlap=False) 
            
            dfFft = rfft(numpyArray) 
            dfFftDescribe=pd.DataFrame(np.array(abs(dfFft)).ravel()).describe()           
            
            freqs, dfPsd = welch(numpyArray, fs=SampleRate, return_onesided=False, detrend=False) 
            dfPsdDescribe = pd.DataFrame(np.array(abs(dfPsd)).ravel()).describe()

            freq_res = freqs[1] - freqs[0]
             
            low1, high1 = 0,0.5 #Compute the average bandpower             
            idx_low = np.logical_and(freqs >= low1, freqs <= high1) # Find intersecting values in frequency vector
            abpLow = simpson(dfPsd[idx_low], dx=freq_res)

            low2, high2 = 0.5, 4            
            idx_delta = np.logical_and(freqs >= low2, freqs <= high2) # Find intersecting values in frequency vector
            abpDelta = simpson(dfPsd[idx_delta], dx=freq_res)

            low3, high3 = 4, 8           
            idx_theta = np.logical_and(freqs >= low3, freqs <= high3) # Find intersecting values in frequency vector
            abpTheta = simpson(dfPsd[idx_theta], dx=freq_res)

            low4, high4 = 8, 12            
            idx_alfa = np.logical_and(freqs >= low4, freqs <= high4) # Find intersecting values in frequency vector
            abpAlpha = simpson(dfPsd[idx_alfa], dx=freq_res)
          
            low5, high5 = 12, 30            
            idx_beta = np.logical_and(freqs >= low5, freqs <= high5) # Find intersecting values in frequency vector
            abpBeta = simpson(dfPsd[idx_beta], dx=freq_res)

            dataQ_describe= pd.DataFrame(np.array(abs(dataQ)).ravel()).describe()
            
            if stage == False: obj = [ classe, nameClass, cminu]
            else: obj = [ classe, nameClass, cminu]
            obj += [
                    float(abs(apenC)), float(abs(dfaC)), float(abs(abpLow)),float(abs(abpDelta)),float(abs(abpTheta)), float(abs(abpAlpha)),  float(abs(abpBeta)),       
                    float(dfFftDescribe.iloc[1][0]), float(abs(dfFftDescribe.iloc[3][0])), float(abs(dfFftDescribe.iloc[-1][0])), float(abs(dfFftDescribe.iloc[2][0])), 
                    float(abs(dfPsdDescribe.iloc[1][0])), float(abs(dfPsdDescribe.iloc[3][0])), float(abs(dfPsdDescribe.iloc[-1][0])), float(abs(dfPsdDescribe.iloc[2][0])), 
                    float(abs(dataQ_describe.iloc[1][0])), float(abs(dataQ_describe.iloc[3][0])), float(abs(dataQ_describe.iloc[-1][0])), float(abs(dataQ_describe.iloc[2][0])) 
                  ]

            results.append(obj)

            if timeStart != 0: cminu=self._setTimeMinu(cminu,minCut)
            else:cminu += minCut
            
    return results

  def _setTimeMinu(self, cminu, minuq):
    """
    Parameters
    ----------
    cminu: int
           Time used for TDAF analysis  
    minuq: int
           The time to be added

    """
    cminu += minuq
    if cminu > 2359: cminu=0
    elif cminu > 60 and cminu < 100: cminu=100
    elif cminu > 160 and  cminu < 200: cminu=200
    elif cminu > 260 and  cminu < 300: cminu=300
    elif cminu > 360 and  cminu < 400: cminu=400
    elif cminu > 460 and  cminu < 500: cminu=500
    elif cminu > 560 and  cminu < 600: cminu=600
    elif cminu > 660 and  cminu < 700: cminu=700
    elif cminu > 760 and  cminu < 800: cminu=800
    elif cminu > 860 and  cminu < 900: cminu=900
    elif cminu > 960 and  cminu < 1000: cminu=1000
    elif cminu > 1060 and  cminu < 1100: cminu=1100
    elif cminu > 1160 and  cminu < 1200: cminu=1200
    elif cminu > 1260 and  cminu < 1300: cminu=1300
    elif cminu > 1360 and  cminu < 1400: cminu=1400
    elif cminu > 1460 and  cminu < 1500: cminu=1500
    elif cminu > 1560 and  cminu < 1600: cminu=1600
    elif cminu > 1660 and  cminu < 1700: cminu=1700
    elif cminu > 1760 and  cminu < 1800: cminu=1800
    elif cminu > 1860 and  cminu < 1900: cminu=1900
    elif cminu > 1960 and  cminu < 2000: cminu=2000
    elif cminu > 2060 and  cminu < 2100: cminu=2100
    elif cminu > 2160 and  cminu < 2200: cminu=2200
    return cminu
      
  
  def salveDf(self, results, classe, stage=False):  
    if stage == False: colmns = ['classe', 'nameClass', 'minuted']
    else: colmns = ['classe', 'nameClass', 'minuted','stage']
    colmns += ['apen', 'dfa','abp_low','abp_delta','abp_theta', 'abp_alpha','abp_beta',
           'fft_mean', 'fft_min', 'fft_max', 'fft_variance', 
           'psd_mean', 'psd_min', 'psd_max' , 'psd_variance', 
           'electrome_mean', 'electrome_min', 'electrome_max', 'electrome_variance', 
          ]
    data = pd.DataFrame(results, columns = colmns)
          
    pathDf = self.pathFeatures+'\\df'+str(classe)+'.csv'
    try:
        data.to_csv(pathDf)
        return data
    except Exception as er:
        return er

class Filter: 
        
    def __init__(self, HPF=0.5, LPF=32,  order=4):
        """
        Parameters
        ----------
        HPF : float
              The critical frequency or frequencies. For lowpass and highpass filters, 
        LPF : float
              The critical frequency or frequencies. For lowpass and highpass filters,
        order: int
               The order of the filter.
        """
        global  GfilePathR, GfilePathF     

        self.pathRaw = GfilePathR
        self.pathFiltered = GfilePathF 
        self.lowcut = np.float16(HPF)
        self.highcut = np.int8(LPF)
        self.order= np.int8(order)

    def starFilter(self, dataP, pathC, fileN, SampleRate=250,  NOTCH=60, quality=10,   show=False): 
        """ Data filtering.
        It should be used for electrophysiological series that have not gone through any previous filtering standard. 
        For example, series captured by OpenBCI or some other prototype. In our tests, the filtering of the series 
        captured from the Mp36 equipment did not cause losses. But, it is not recommended to use it in this case.

        Parameters
        ----------
        dataP : ndarray
                 The array of data to be filtered.
        pathC : str
                Path to new file
        fileN : str
                 Name to new file
        SampleRate: int
                    capture frequency
        NOTCH : float
                Frequency to remove from a signal. In Brazil 60Hz is used
        quality : float
                  Quality factor. Dimensionless parameter that characterizes
                  notch filter -3 dB bandwidth ``bw`` relative to its center
                  frequency, ``Q = w0/bw``.
       
        show:  bool
               if true it will plot the signal graphs
        """
        if isinstance(dataP, (np.ndarray, np.generic)):dataN=dataP
        else:dataN=np.float64(dataP.to_numpy()[:,0].ravel())   
        
        fs = np.int16(SampleRate)  
        f0 = np.int8(NOTCH) 
        Q  = np.float16(quality)  
        w0 = np.float16(f0 / (fs / 2))  # normalized frequency
       
        h = self._notch_filter(dataN, w0, Q)
        dataF = self._butter_bandpass_filter(h, fs) 
        saveFile=self._save_file(dataF, fileN, pathC)
        if saveFile:
            if show:
                dataP['MicroVF']=dataF
                plots=_Plots()
                plots.plotFig1(dataN, dataF, dataP, fileN)
            return [np.array(dataN),np.array(dataF),np.array(dataP)]
        else:
            print('Error saving filtered signal: '+str(saveFile)) 
            os._exit(0)


    def _notch_filter(self, data, w0, Q):
        """
        A notch filter is a band-stop filter with a narrow bandwidth
        (high quality factor). It rejects a narrow frequency band and
        leaves the rest of the spectrum little changed.

        Parameters
        ----------
        data : array_like
               The array of data to be filtered.
        w0 : float
            Frequency to remove from a signal. 
        Q : float
            Quality factor. Dimensionless parameter that characterizes
            notch filter -3 dB bandwidth ``bw`` relative to its center
            frequency, ``Q = w0/bw``.

        Returns
        -------
        h : ndarray
            The filtered output with the same shape as `data`.

       
        """
        
        b, a = iirnotch(w0, Q)
        h = np.array(filtfilt(b, a, data), dtype=np.float64) 
        
        return h
    

    def _butter_bandpass_filter(self, h, fs):
        """
        Butterworth digital and analog filter design.

        Design an Nth-order digital Butterworth filter and return
        the filter coefficients, after the filtfilt method returns the filtered signal
        
        Parameters
        ----------
        h : array_like
            The array of data to be filtered.
        fs : float
            The sampling frequency of the digital system.
        
        Returns
        -------
        y : ndarray
            The filtered output with the same shape as `h`.
        """
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='bandpass')
        y = np.array(filtfilt(b, a, h), dtype=np.float64) 
        
        return y
    
    

  
    def _save_file(self, dataN, fileN, pathC):
        """
        Save the filtered signal

        Parameters
        ----------
        dataN : ndarray
                 The array of data to be saved.
        fileN : str
                 Name to new file
        pathC  : str
                 Path to new file

        """
        pathsave = os.path.join(self.pathFiltered,pathC, fileN) 
        try:
            np.savetxt(pathsave+'.txt', dataN, fmt='%1.9f')  
            return True
        except Exception as er:
            return er
    
class Plots:

    def __init__(self, SMALL_SIZE = 12, MEDIUM_SIZE = 14, BIGGER_SIZE = 20, FONT_SIZE = 8) -> None:
        global  GfilePathR, GfilePathF, GfilePathG, GfilePathFe

        self.pathGraph = GfilePathG
        self.pathRaw = GfilePathR
        self.pathFeature = GfilePathFe
        self.pathFilter = GfilePathF

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=FONT_SIZE)     # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # title fontsize
        
        sns.set(font_scale=1.5, style='white')

    def plotFilter(self, dataN=0, dataF=0):
        fig = plt.figure(figsize=(15,5), constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        ax1 = fig.add_subplot(spec2[0, 0])
        ax2 = fig.add_subplot(spec2[0, 1])

        sns.lineplot( x=range(0,len(dataN)), y=dataN, ax=ax1, palette='terrain')
        sns.lineplot( x=range(0,len(dataF)), y=dataF, ax=ax2, palette='terrain')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax1.set_title('raw Signal')
        ax2.set_title('filtered Signal')

        ax1.set_xlabel('Points')
        ax2.set_xlabel('Points')

       
        ax1.set_ylabel('uV')
        ax2.set_ylabel('uV') 
        pthg=os.path.join(self.pathGraph, 'raw_filtered.png') 
        
        plt.savefig(pthg,dpi=500, transparent=False, bbox_inches='tight')
        plt.show()
    
    def plotTDAF(self, filepathC,text_box, sample_rate=50):
        all_files = glob.glob(self.pathFeature + "/*.csv")
        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        dados = pd.concat(li, axis=0, ignore_index=True)

        title1='TDAF-electrome'
        title2='TDAF-FFT'
        text_box.insert(1.0, "Creating Ploat "+title1+" "+title2+". Please wait.\n")  
        text_box.pack()

        fig = plt.figure(constrained_layout=True, figsize=(15,12))
        spec = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[2, 0])
        ax4 = fig.add_subplot(spec[3, 0])

        ax5 = fig.add_subplot(spec[0, 1])
        ax6 = fig.add_subplot(spec[1, 1])
        ax7 = fig.add_subplot(spec[2, 1])
        ax8 = fig.add_subplot(spec[3, 1])

        
        sns.lineplot(data=dados, x="minuted", y="electrome_mean",hue="nameClass",  ax=ax1,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)  
        sns.lineplot(data=dados, x="minuted", y="electrome_min",hue="nameClass", ax=ax2,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="electrome_max",hue="nameClass", ax=ax3,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="electrome_variance",hue="nameClass", ax=ax4,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)     

        sns.lineplot(data=dados, x="minuted", y="fft_mean",hue="nameClass", ax=ax5,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)  
        sns.lineplot(data=dados, x="minuted", y="fft_min",hue="nameClass", ax=ax6,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="fft_max",hue="nameClass", ax=ax7,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="fft_variance",hue="nameClass", ax=ax8,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)   

        plt.setp(ax1, title=title1)    
        plt.setp(ax5, title=title2)      
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['top'].set_visible(False)

        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('')
        ax4.set_xlabel('Time')
        ax5.set_xlabel('')
        ax6.set_xlabel('')
        ax7.set_xlabel('')
        ax8.set_xlabel('Time')

        ax1.legend(loc='upper right', framealpha=0.1, title=None)
        ax2.get_legend().remove()
        ax3.get_legend().remove()
        ax4.get_legend().remove()
        ax5.legend(loc='upper right', framealpha=0.1, title=None)
        ax6.get_legend().remove()
        ax7.get_legend().remove()
        ax8.get_legend().remove()
        try:
            pathG = os.path.join(self.pathGraph, title1+'_'+title2+'.png')        
            plt.savefig(pathG,dpi=500)
            text_box.insert(1.0, 'Ploat saved.\n')  
            text_box.pack()  
        except:
            return True
        
        fig = plt.figure(constrained_layout=True, figsize=(15,12))
        spec = gridspec.GridSpec(ncols=2, nrows=5, figure=fig)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[2, 0])
        ax4 = fig.add_subplot(spec[3, 0])

        ax5 = fig.add_subplot(spec[0, 1])
        ax6 = fig.add_subplot(spec[1, 1])
        ax7 = fig.add_subplot(spec[2, 1])
        ax8 = fig.add_subplot(spec[3, 1])
        ax9 = fig.add_subplot(spec[4, 1])

        title1='TDAF-PSD'
        title2='TDAF-ABP'
        text_box.insert(1.0, "Creating Ploat "+title1+" "+title2+". Please wait.\n")  
        text_box.pack()

        sns.lineplot(data=dados, x="minuted", y="psd_mean",hue="nameClass", ax=ax1,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)  
        sns.lineplot(data=dados, x="minuted", y="psd_min",hue="nameClass", ax=ax2,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="psd_max",hue="nameClass", ax=ax3,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="psd_variance",hue="nameClass", ax=ax4,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)     

        sns.lineplot(data=dados, x="minuted", y="abp_low",hue="nameClass", ax=ax5,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)  
        sns.lineplot(data=dados, x="minuted", y="abp_delta",hue="nameClass", ax=ax6,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="abp_theta",hue="nameClass", ax=ax7,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11) 
        sns.lineplot(data=dados, x="minuted", y="abp_alpha",hue="nameClass", ax=ax8,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11) 
        sns.lineplot(data=dados, x="minuted", y="abp_beta",hue="nameClass", ax=ax9,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11) 



        plt.setp(ax1, title=title1)    
        plt.setp(ax5, title=title2)      
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False) 
        ax4.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['top'].set_visible(False)
        ax9.spines['right'].set_visible(False)
        ax9.spines['top'].set_visible(False)

        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('')
        ax4.set_xlabel('Time')
        ax5.set_xlabel('')
        ax6.set_xlabel('')
        ax7.set_xlabel('')
        ax8.set_xlabel('')
        ax9.set_xlabel('Time')

        ax1.legend(loc='upper right', framealpha=0.1, title=None)
        ax2.get_legend().remove()
        ax3.get_legend().remove()
        ax4.get_legend().remove()
        ax5.legend(loc='upper right', framealpha=0.1, title=None)
        ax6.get_legend().remove()
        ax7.get_legend().remove()
        ax8.get_legend().remove()
        ax9.get_legend().remove()

        try:
            pathG = os.path.join(self.pathGraph, title1+'_'+title2+'.png')        
            plt.savefig(pathG,dpi=500)
            text_box.insert(1.0, 'Ploat saved.\n')  
            text_box.pack()  
        except:
            return True

        fig = plt.figure(constrained_layout=True, figsize=(15,7))
        spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])


        title1='TDAF-ApEn'
        title2='TDAF-DFA'
        text_box.insert(1.0, "Creating Ploat "+title1+" "+title2+". Please wait.\n")  
        text_box.pack()
        sns.lineplot(data=dados, x="minuted", y="apen",hue="nameClass", ax=ax1,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)  
        sns.lineplot(data=dados, x="minuted", y="dfa",hue="nameClass", ax=ax2,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11) 

        plt.setp(ax1, title=title1)    
        plt.setp(ax2, title=title2)      
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax1.set_xlabel('Time')
        ax2.set_xlabel('Time')

        ax1.legend(loc='upper right', framealpha=0.1, title=None)
        ax2.legend(loc='upper right', framealpha=0.1, title=None)


        try:
            pathG = os.path.join(self.pathGraph, title1+'_'+title2+'.png')        
            plt.savefig(pathG,dpi=500)
            text_box.insert(1.0, 'Ploat saved.\n')  
            text_box.pack()  
        except:
            return True

        return False
    
    def plotTDAFLog(self, filepathC,text_box, sample_rate=50):
        all_files = glob.glob(self.pathFeature + "/*.csv")
        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        dados = pd.concat(li, axis=0, ignore_index=True)

        title1='TDAF-electrome'
        title2='TDAF-FFT'
        text_box.insert(1.0, "Creating Ploat "+title1+" "+title2+". Please wait.\n")  
        text_box.pack()

        fig = plt.figure(constrained_layout=True, figsize=(15,12))
        spec = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[2, 0])
        ax4 = fig.add_subplot(spec[3, 0])

        ax5 = fig.add_subplot(spec[0, 1])
        ax6 = fig.add_subplot(spec[1, 1])
        ax7 = fig.add_subplot(spec[2, 1])
        ax8 = fig.add_subplot(spec[3, 1])

        
        sns.lineplot(data=dados, x="minuted", y="electrome_mean",hue="nameClass", ax=ax1,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)  
        sns.lineplot(data=dados, x="minuted", y="electrome_min",hue="nameClass", ax=ax2,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="electrome_max",hue="nameClass", ax=ax3,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="electrome_variance",hue="nameClass", ax=ax4,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)     

        sns.lineplot(data=dados, x="minuted", y="fft_mean",hue="nameClass", ax=ax5,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)  
        sns.lineplot(data=dados, x="minuted", y="fft_min",hue="nameClass", ax=ax6,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="fft_max",hue="nameClass", ax=ax7,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="fft_variance",hue="nameClass", ax=ax8,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)   

        plt.setp(ax1, title=title1)    
        plt.setp(ax5, title=title2)      
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['top'].set_visible(False)

        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('')
        ax4.set_xlabel('Time')
        ax5.set_xlabel('')
        ax6.set_xlabel('')
        ax7.set_xlabel('')
        ax8.set_xlabel('Time')

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax5.set_yscale('log')
        ax6.set_yscale('log')
        ax7.set_yscale('log')
        ax8.set_yscale('log')
        

        ax1.legend(loc='upper right', framealpha=0.1, title=None)
        ax2.get_legend().remove()
        ax3.get_legend().remove()
        ax4.get_legend().remove()
        ax5.legend(loc='upper right', framealpha=0.1, title=None)
        ax6.get_legend().remove()
        ax7.get_legend().remove()
        ax8.get_legend().remove()
        try:
            pathG = os.path.join(self.pathGraph, title1+'_'+title2+'_Log.png')        
            plt.savefig(pathG,dpi=500)
            text_box.insert(1.0, 'Ploat saved.\n')  
            text_box.pack()  
        except:
            return True
        
        fig = plt.figure(constrained_layout=True, figsize=(15,12))
        spec = gridspec.GridSpec(ncols=2, nrows=5, figure=fig)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[2, 0])
        ax4 = fig.add_subplot(spec[3, 0])

        ax5 = fig.add_subplot(spec[0, 1])
        ax6 = fig.add_subplot(spec[1, 1])
        ax7 = fig.add_subplot(spec[2, 1])
        ax8 = fig.add_subplot(spec[3, 1])
        ax9 = fig.add_subplot(spec[4, 1])

        title1='TDAF-PSD'
        title2='TDAF-ABP'
        text_box.insert(1.0, "Creating Ploat "+title1+" "+title2+". Please wait.\n")  
        text_box.pack()

        sns.lineplot(data=dados, x="minuted", y="psd_mean",hue="nameClass", ax=ax1,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)  
        sns.lineplot(data=dados, x="minuted", y="psd_min",hue="nameClass", ax=ax2,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="psd_max",hue="nameClass", ax=ax3,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="psd_variance",hue="nameClass", ax=ax4,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)     

        sns.lineplot(data=dados, x="minuted", y="abp_low",hue="nameClass", ax=ax5,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)  
        sns.lineplot(data=dados, x="minuted", y="abp_delta",hue="nameClass", ax=ax6,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11)
        sns.lineplot(data=dados, x="minuted", y="abp_theta",hue="nameClass", ax=ax7,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11) 
        sns.lineplot(data=dados, x="minuted", y="abp_alpha",hue="nameClass", ax=ax8,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11) 
        sns.lineplot(data=dados, x="minuted", y="abp_beta",hue="nameClass", ax=ax9,dashes= False, style = 'nameClass', palette = 'hot', markers= ["1","2","3","4"],ms=11) 



        plt.setp(ax1, title=title1)    
        plt.setp(ax5, title=title2)      
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False) 
        ax4.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['top'].set_visible(False)
        ax9.spines['right'].set_visible(False)
        ax9.spines['top'].set_visible(False)

        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('')
        ax4.set_xlabel('Time')
        ax5.set_xlabel('')
        ax6.set_xlabel('')
        ax7.set_xlabel('')
        ax8.set_xlabel('')
        ax9.set_xlabel('Time')

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax5.set_yscale('log')
        ax6.set_yscale('log')
        ax7.set_yscale('log')
        ax8.set_yscale('log')
        ax9.set_yscale('log')

        ax1.legend(loc='upper right', framealpha=0.1, title=None)
        ax2.get_legend().remove()
        ax3.get_legend().remove()
        ax4.get_legend().remove()
        ax5.legend(loc='upper right', framealpha=0.1, title=None)
        ax6.get_legend().remove()
        ax7.get_legend().remove()
        ax8.get_legend().remove()
        ax9.get_legend().remove()

        try:
            pathG = os.path.join(self.pathGraph, title1+'_'+title2+'_Log.png')        
            plt.savefig(pathG,dpi=500)
            text_box.insert(1.0, 'Ploat saved.\n')  
            text_box.pack()  
        except:
            return True   

        return False

    def plotFftPsd(self, filepathC,text_box, sample_rate=50):
        for pathC in filepathC:
            pathFC = os.path.join(self.pathRaw, pathC) 
            
            directory = os.listdir(pathFC)
            
            pathFilterFile = os.path.join(self.pathFilter, pathC) 
            
            for file in directory:
                
                    fileN, exte  = os.path.splitext(str(file))

                    if(exte == '.csv' or exte == '.txt'):
                        text_box.insert(1.0, "Creating Ploat to "+fileN+". Please wait.\n")  
                        text_box.pack()
                        pathFileFinal = os.path.join(pathFC, file) 
                        dataP = pd.read_csv(pathFileFinal, encoding = 'utf-8', 
                                                delimiter=",", 
                                                engine='c', 
                                                low_memory=False, 
                                                memory_map=True,
                                                )
                        pathFileFinalFilter = os.path.realpath(os.path.join(pathFilterFile, fileN+'.txt'))
                        dataF = pd.read_csv(pathFileFinalFilter, encoding = 'utf-8', 
                                                delimiter=",", 
                                                engine='c', 
                                                low_memory=False, 
                                                memory_map=True,
                                                )
                        
                       
                        dataP=np.float64(dataP.to_numpy()[:len(dataF),0].ravel()) 
                        dataF=np.float64(dataF.to_numpy().ravel())
                        dataX=range(0, len(dataP))
                        
                        # Define window length (4 seconds)
                        win = 4 * sample_rate
                        dataFft = np.abs(rfft(dataF))
                        freqs_fft = np.abs(rfftfreq(dataF.shape[-1], win))*10000
                        
                        freqs_psd, dataPsd = welch(dataF, fs=sample_rate, nperseg=win, return_onesided=False, detrend=False) 
                        freqs_psd= np.abs(freqs_psd)

                        fig = plt.figure(figsize=(15,5), constrained_layout=True)
                        spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
                        ax1 = fig.add_subplot(spec2[0, 0])
                        ax2 = fig.add_subplot(spec2[0, 1])
                        ax3 = fig.add_subplot(spec2[1, 0])
                        ax4 = fig.add_subplot(spec2[1, 1])

                        sns.lineplot( y=dataP, x=dataX, ax=ax1, palette='terrain')
                        sns.lineplot( y=dataF, x=dataX, ax=ax2, palette='terrain')
                        sns.lineplot( x=freqs_fft, y=dataFft, ax=ax3, palette='terrain')
                        sns.lineplot( x=freqs_psd, y=dataPsd, ax=ax4, palette='terrain')
                        
                        
                        '''xlabs=[]
                        for i in ax1.get_xticks():
                            i='{0:.0f}'.format(i)
                            if i == '0': xlabs.append('00:00')
                            elif len(i) > 3:
                                if str(i[2]+i[3]) == '60': xlabs.append(i[0]+i[1]+':59') 
                                else:xlabs.append(i[0]+i[1]+':'+i[2]+i[3])           
                            else: xlabs.append('')
                        ax1.set_xticklabels(xlabs, rotation=25)

                        xlabs=[]
                        for i in ax2.get_xticks():
                            i='{0:.0f}'.format(i)
                            if i == '0': xlabs.append('00:00')
                            elif len(i) > 3:
                                if str(i[2]+i[3]) == '60': xlabs.append(i[0]+i[1]+':59') 
                                else:xlabs.append(i[0]+i[1]+':'+i[2]+i[3])          
                            else: xlabs.append('')      
                        ax2.set_xticklabels(xlabs, rotation=25)'''

                        ax1.spines['right'].set_visible(False)
                        ax1.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['top'].set_visible(False)
                        ax3.spines['right'].set_visible(False)
                        ax3.spines['top'].set_visible(False)
                        ax4.spines['right'].set_visible(False)
                        ax4.spines['top'].set_visible(False)

                        ax1.set_title('Raw Signal')
                        ax2.set_title('Filtered Signal')
                        

                        ax1.set_xlabel('Time')
                        ax2.set_xlabel('Time')
                        ax3.set_xlabel('Frequency (Hz)')
                        ax4.set_xlabel('Frequency (Hz)')

                        ax1.set_ylabel('uV')
                        ax2.set_ylabel('uV')
                        ax3.set_ylabel('FFT')
                        ax4.set_ylabel('PSD ')
                        try:
                            pthg=os.path.join(self.pathGraph, 'figure_'+fileN+'.png') 
                            plt.savefig(pthg,dpi=500, transparent=False, bbox_inches='tight')
                            text_box.insert(1.0, 'Ploat saved.\n')  
                            text_box.pack()                            
                        except:
                            return True
        return False
    
    

    def plotMatriz(self, matriz, nameClass, pasta='_'):
      df = pd.DataFrame(matriz, columns= nameClass)
      df['y'] = nameClass
      df = df.set_index('y')

      mUpper = np.triu(df)
      mask = np.triu(np.ones_like(df, dtype=np.bool))

      f, ax = plt.subplots(figsize=(10, 8))
      ax = sns.heatmap(df, annot=True, fmt="3.0f", vmin=0, vmax=100, cmap='winter' ) #ocean
      plt.savefig(self.G+'\\'+str(pasta)+'_confusao_Accuracy.png', dpi=300)
      plt.show()