# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 19:11:44 2016
This tollBox is used to create loader for all sort of data
@author: Alberto
"""
import sys
import numpy as np
import scipy.io as sio
import os.path
from ast import literal_eval


def GenCodeList_Info(FileName):
    '''Load information from GenCode list, according to its ionner format
    Input
       FileName: Name of the file, including full path
    Outputs
       Name: Name of the gencode
       Sort: Sort of excitation, text ('chirp','pulse','burst')
       ProgGenCLKfreqMHz: ProgGenCLKfreqMHz, float
       F1: Lower (chirp) or central (burst, pulse) frequency in MHz, float
       F2: higher (chirp) or central (burst, pulse) frequency in MHz, float
       Cycles: number of cycles of the burst, float
       Duration: Duration of the signal, us, float
       Polarity: Polarity of the pulse (-1, 1, 2), integer    
    '''
    try:
        GenCodeList = np.loadtxt(FileName, dtype={'names': ('Name', 'Sort', 'Fclk', 'F1','F2','NCyc','Dur','Pol'),
                     'formats': ('S10', 'S5','f','f','f','f','f','int')}, delimiter=';')
        titulo=[]
        for GNo in range(GenCodeList.size):
            if GenCodeList[GNo][1]=='chirp':
                patata=str(GenCodeList[GNo][0]) + ': ' + str(GenCodeList[GNo][1]) + ' BW(MHz)=[' + str(GenCodeList[GNo][3]) + ',' +\
                str(GenCodeList[GNo][4]) + '] ' + u'\u0394\u03C4=' + str(GenCodeList[GNo][6]) + u'\u03BCs'
            
            elif GenCodeList[GNo][1]=='burst':
                patata = str(GenCodeList[GNo][0]) + ': ' + str(GenCodeList[GNo][1]) + ' Fc=' + str(GenCodeList[GNo][3]) + \
                'MHz NoCycles=' + str(GenCodeList[GNo][5]) + u' \u0394\u03C4=' + str(GenCodeList[GNo][6]) + u'\u03BCs'
            else:
                patata = str(GenCodeList[GNo][0]) + ': ' + str(GenCodeList[GNo][1]) + ' Fc=' + str(GenCodeList[GNo][3])  +\
                'MHz ' + u'\u0394\u03C4=' + str(GenCodeList[GNo][6]) + u'\u03BCs'
            titulo.append(patata)
        return GenCodeList, titulo
    except Exception as e:
        print(e)
        return ''

def loadFromMatlab(FileName, DataName):
    """
    Load file from matlab to numpy array.

    Parameters
    ----------
    FileName : str, full name of file including extension and path
    DataName : str, name of the data stored in .mat file

    Returns
    -------
    DataMatrix, np.array of data.

    """
    mat_contents = sio.loadmat(FileName, squeeze_me=True, mat_dtype=True)
    # a = mat_contents.keys()
    DataMatrix = mat_contents[DataName]
    return DataMatrix


# this creates variables with all the variables of acquisition in standard.var
class StdVar:
    def __init__(self, FileName):
        
        with open(FileName, 'r') as f:
            for line in f: # search in all lines
#                print line
                [A,B] = line.split(' - ')
                [C,D] = B.split('\n')
                try: # check if they are numeric
                    setattr(self,C,int(A)) # create new attribute as integer
                except ValueError:
                    try:
                        setattr(self,C,float(A)) # create new attribute as float
                    except ValueError:
                        setattr(self,C,A) # create new attribute as string
        f.close()
        

# Load data from bin file and reshape into matrix. USed for Cscans
def LoadBinCscan(filename, Xsteps, Ysteps, Avg, ScanLen, N1=0, N2=0):
    """
    Load Cscan from Bin file, lithuanian ACQ format
    Inputs:
        filename = file to load, *.bin
        Xsteps number of scanning points along X axis for Cscan
        Ysteps number of scanning points along Y axis for Cscan
        Avg = number of Ascans acq. at each location, to be averaged
        ScanLen = total scanlength
        N1 = starting sample for the output
        N2 = last sample for the output, if 0 then to ScanLength
    Outputs:
        MyData = 3D matrix with Cscan [Xsteps,ScanLen,Ysteps]

    """
    with open(filename, 'rb') as fd:
        if N2 == 0:
            N2 = ScanLen
        MyData = np.zeros((Xsteps, N2-N1, Ysteps))
    
        if Avg == 1:
            for i in range(Xsteps):
                for j in range(Ysteps):
                    Ascan = np.fromfile(fd, dtype=np.uint16, count=ScanLen)/1024.
                    if i%2==0:  
                        MyData[i,:,j] = (Ascan - np.mean(Ascan))[N1:N2]
                    else:
                        MyData[i,:,Ysteps-j-1] = (Ascan - np.mean(Ascan))[N1:N2]
    
        else:
            for i in range(Xsteps):
                for j in range(Ysteps):
                    Ascan = np.zeros(ScanLen)
                    for k in range(Avg):
                        Ascan = np.fromfile(fd, dtype=np.uint16, count=ScanLen) + Ascan
                    Ascan = (Ascan/1024.) / Avg
                    if i%2==0:  
                        MyData[i,:,j] = (Ascan - np.mean(Ascan))[N1:N2]
                    else:
                        MyData[i,:,Ysteps-j-1] = (Ascan - np.mean(Ascan))[N1:N2]
    return MyData

# Load data from bin file and reshape into matrix. USed for Cscans
def LoadBinCscanFFT(filename, Xsteps, Ysteps, Avg, ScanLen, N1=0, N2=0, Fs=100e6, nfft=1024,Flim=10):
    """
    Load Cscan from Bin file, lithuanian ACQ format
    Inputs:
        filename = file to load, *.bin
        Xsteps number of scanning points along X axis for Cscan
        Ysteps number of scanning points along Y axis for Cscan
        Avg = number of Ascans acq. at each location, to be averaged
        ScanLen = total scanlength
        N1 = starting sample for the output
        N2 = last sample for the output, if 0 then to ScanLength
    Outputs:
        MyData = 3D matrix with Cscan [Xsteps,ScanLen,Ysteps]
        
    """

    N = int(Flim/Fs*nfft)
    with open(filename, 'rb') as fd:
        if N2==0:        
            N2=ScanLen
        MyData = np.zeros((Xsteps,N,Ysteps))
        if Avg==1:
            for i in range(Ysteps):            
                for j in range(Xsteps):
                    Ascan = np.fromfile(fd, dtype=np.uint16, count=ScanLen)/1024.
                    Bscan = (Ascan - np.mean(Ascan))[N1:N2]
                    MyData[j,:,i] = np.abs(np.fft.fft(Bscan,nfft))[:N]
                    
        else:
            for i in range(Ysteps):
                for j in range(Xsteps):
                    Ascan = np.zeros(ScanLen)
                    for k in range(Avg):
                        Ascan = np.fromfile(fd, dtype=np.uint16,count=ScanLen) + Ascan
                    Ascan = (Ascan/1024.) / Avg
                    Bscan = (Ascan - np.mean(Ascan))[N1:N2]
                    MyData[j,:,i] = np.abs(np.fft.fft(Bscan,nfft))[:N]                
    return MyData

# Load data from bin file and reshape into matrix. USed for bscans
def LoadBinBscan(filename, Xsteps, Avg, ScanLen, N1=0, N2=0):
    """
    Load Bscan from Bin file, lithuanian ACQ format
    Inputs:
        filename = file to load, *.bin
        Xsteps number of scanning points for Bscan
        Avg = number of Ascans acq. at each location, to be averaged
        ScanLen = total scanlength
        N1 = starting sample for the output
        N2 = last sample for the output, if 0 then to ScanLength
    Outputs:
        MyData = 2D matrix with Cscan [Xsteps,ScanLen]        
    """
    with open(filename, 'rb') as fd:
        if N2==0:        
            N2=ScanLen
        MyData = np.zeros((Xsteps,N2-N1))
        if Avg==1:
            for j in range(Xsteps):            
                Ascan = np.fromfile(fd, dtype=np.uint16, count=ScanLen)/1024.
                MyData[j,:] = (Ascan - np.mean(Ascan))[N1:N2]                
        else:
            for j in range(Xsteps):
                Ascan = np.zeros(ScanLen)
                for k in range(Avg):
                    Ascan = np.fromfile(fd, dtype=np.uint16, count=ScanLen) + Ascan
                Ascan = (Ascan/1024.) / Avg                
                MyData[j,:] = (Ascan - np.mean(Ascan))[N1:N2]
    return MyData

# Load data from bin file and reshape into matrix. USed for Ascans
def LoadBinAscan(filename, Avg, ScanLen, N1=0, N2=0):
    """
    Load Ascan from Bin file, lithuanian ACQ format
    Inputs:
        filename = file to load, *.bin
        Avg = number of Ascans acq. at each location, to be averaged
        ScanLen = total scanlength
        N1 = starting sample for the output
        N2 = last sample for the output, if 0 then to ScanLength
    Outputs:
        MyData = 2D matrix with Cscan [Xsteps,ScanLen]        
    """
    with open(filename, 'rb') as fd:
        if N2==0:        
            N2=ScanLen
        MyData = np.zeros(N2-N1)
        if Avg==1:        
            Ascan = np.fromfile(fd, dtype=np.uint16, count=ScanLen)/1024.
            MyData = (Ascan - np.mean(Ascan))[N1:N2]                
        else:
            Ascan = np.zeros(ScanLen)
            MyAvg = Avg
            for k in range(Avg):
                Ascan1 = np.fromfile(fd, dtype=np.uint16, count=ScanLen) 
                if not(np.all(Ascan1==0.0)):                
                    Ascan = Ascan1 + Ascan
                else:
                    MyAvg = MyAvg-1
            print('%d Ascans rejected '%(Avg-MyAvg))
            Ascan = (Ascan/1024.) / MyAvg                
            MyData = (Ascan - np.mean(Ascan))[N1:N2]
    return MyData


# create Data_Matrix loading binary files from acquisition
# very old version, maybe not used anymore
class SingleCycleData:
    def __init__(self, FileName, MyStdVar, LoadType = 'SingleCycle',
                 Average = 'True', ChNr = 1, SigNr = 1, LoadRange = np.array([0])):
        # Check simensions
        x = MyStdVar.GenCode # number of excitations
        y = MyStdVar.TestCycNr #number of experiment repetitions
        z = MyStdVar.AvgSamplesNumber # numer of Ascan per step
        s = MyStdVar.Smax-MyStdVar.Smin # number of samples per Ascan
        Data = Load_data_bin(FileName,x,y,z,s)/1024.0 # Load Data
        if LoadRange.size>1:
            Data = Data[:,:,:,LoadRange]
            s = LoadRange.size
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    Data[i,j,k,:] = Data[i,j,k,:]-np.mean(Data[i,j,k,:])
                    
        if Average:
            Data = np.mean(Data,axis=2)
            Data.resize((x,y,s))
        else:
            if z == 1:
                Data.resize((x,y,s))
        shape = Data.shape
        if shape[0] == 1:
            Data.resize(shape[1:])
            shape = Data.shape
            if shape[0] == 1:
                if len(shape)>2:
                    Data.resize(shape[1:])
        else:
            if shape[1] == 1:
                if len(shape)>3:
                    Data.resize(x,z,s)
                else:
                    Data.resize(x,s)
        self.Data = Data            

          
# Load data from bin file and reshape into matrix. USed in SingleCycleData:
# very old version, maybe not used anymore
def Load_data_bin(filename,x,y,z,s):
# x - number of rows, y - number of columns, z - Ascans per step, s - Ascan length
    shape = (x,y,z,s)
    with open(filename, 'rb') as fd:
        data = np.fromfile(file=fd, dtype=(np.uint16)).reshape(shape)
    return data


# Load 2 channel data data from text file. Also loads standar.var
def Load_Ascans_PE_TT(DataPath):
    #----------------------------------------------
    # Load 2 channels Ascan TT and PE, from text file
    # Ascans already averaged and Gain corrected
    # 
    # Inputs:
    #   DataPath = name of the path where the data is saved
    #
    # Outputs
    #   TT_Ascan = Ascan from channel 1 TT
    #   PE_Ascan = Ascan from channel 2 PE
    #   MyVars = standard vars from experiment
    #
    # Alberto, 24/05/2017
    #----------------------------------------------

    #----------------------------------------------
    # load standvar
    #----------------------------------------------
    FileStdVar = DataPath +"\standard.var" #standard.var file  
    MyVars = StdVar(FileName = FileStdVar)
#    variables = MyVars.__dict__.keys() # all the variables  
    
    #----------------------------------------------
    # Load Ascans
    #----------------------------------------------    
    
    File2Load_1 = DataPath + '\Ch1_Ascan.txt' #Ascan from Ch1
    File2Load_2 = DataPath + '\Ch2_Ascan.txt' #Ascan from Ch2
        
    TT_Ascan = np.fromfile(File2Load_1, dtype=float, count=-1, sep='\n') #load channel 1 Ascan
    PE_Ascan = np.fromfile(File2Load_2, dtype=float, count=-1, sep='\n') #load channel 2 Ascan    
    
    TT_Ascan = TT_Ascan / np.power(10,(MyVars.Gain1/20)) # Correct gain
    PE_Ascan = PE_Ascan / np.power(10,(MyVars.Gain2/20)) # correct gain
           
    return TT_Ascan, PE_Ascan, MyVars

# loads Ascan from file
def Load_Ascan(File2Load):
    #----------------------------------------------
    # Load Ascan from text file
    # Ascans already averaged and Gain corrected
    # Inputs:
    #   File2Load = name of the file where the data is saved, including path
    #
    # Outputs
    #   Ascan = Ascan from file
    #
    # Alberto, 24/05/2017
    #----------------------------------------------
    
    #----------------------------------------------
    # Load Ascans
    #----------------------------------------------    

    Ascan = np.fromfile(File2Load, dtype=float, count=-1, sep='\n') #load channel 2 Ascan    
    
#    TT_Ascan = TT_Ascan / np.power(10,(MyVars.Gain1/20)) # Correct gain
#    PE_Ascan = PE_Ascan / np.power(10,(MyVars.Gain2/20)) # correct gain
           
    return Ascan


def loadAscanBatches(BATCH_DICT, DataPath, lettersList, GenCode=1, Avg=1):
    '''
    Load several batches, each one with several experiments. Folder names of
    experiments must agree with the following nomenclature:
        "X_Experiment"
    where X is a capital letter of the alphabet. Water Path folders for each
    batch must have the same name: "WP". Batches' folder names do not need to
    follow any naming rules.
    
    If batches do not have the same number of experiments, the resulting
    matrices' empty slots will be filled with NaN values.
    
    It is assumed that Ascans where accumulated in acq. stage, but not divided 
    by the number of averages. Therefore, data is divided by AvgSamplesNumber 
    from standard.var.

    Parameters
    ----------
    BATCH_DICT : dictionary
        'Name of batch' : NSpecimens in batch.
    DataPath : string
        Absolute path of batches' folders.
    lettersList : list of strings
        List of capital letters with backslash, i.e., '\A', '\B', '\C', etc.
        These identify each esperiment inside a batch.
    GenCode : int, optional
        Excitation to be loaded. The default is 1.
        1: pulse, 2: burst, 3: chirp
    Avg : int, optional
        Number of Ascans to be read and averaged. The default is 1.

    Returns
    -------
    PE : 3D-matrix
        Pulse Echo signal matrix. PE[batch, specimen, sample]
    TT : 3D-matrix
        Through Transmission signal matrix. TT[batch, specimen, sample]
    WP : 2D-matrix
        Water Path signal matrix. One row is the water path of one batch.

    Arnau, 01/10/2022
    '''
    Max_NSpecimens = len(lettersList)
    NBatches = len(BATCH_DICT)

    # Assuming the same for all experiments
    stdVar = StdVar(DataPath + list(BATCH_DICT.keys())[0] + '\A' + '_Experiment' + r"\standard.var")
    ScanLen = int(stdVar.Smax-stdVar.Smin)

    # beware!! in this experiment, Ascans where accumulated in acq. stage, but
    # not divided by the number of averages, that must be extracted from stdVar
    AvgSamplesNumber = stdVar.AvgSamplesNumber

    PE = np.empty((NBatches, Max_NSpecimens, ScanLen))*np.nan # matrix with all data for PE
    TT = PE.copy() # matrix with all data for TT
    WP = np.empty((NBatches, ScanLen))*np.nan # matrix with all data for WP
    
    for i, (batch, NSpecimens) in enumerate(BATCH_DICT.items()):
        WP_Path = DataPath + batch + '\WP'
        for j in range(NSpecimens):
            Specimen_Path = os.path.abspath(DataPath + batch + lettersList[j] + '_Experiment')
            
            # Load PE
            filename_Ch2 = os.path.abspath(Specimen_Path + r'\ScanADCgenCod' + str(GenCode) + 'Ch2.bin') # load PE
            PE[i, j, :] = LoadBinAscan(filename_Ch2, Avg, ScanLen, N1=0, N2=0) / AvgSamplesNumber

            # Load TT
            filename_Ch1 = os.path.abspath(Specimen_Path + r'\ScanADCgenCod' + str(GenCode) + 'Ch1.bin') # load TT 
            TT[i, j, :] = LoadBinAscan(filename_Ch1, Avg, ScanLen, N1=0, N2=0) / AvgSamplesNumber
            
        # Load WP
        filename_WP = os.path.abspath(WP_Path + r'\ScanADCgenCod' + str(GenCode) + 'Ch1.bin') # load WP
        WP[i, :] = LoadBinAscan(filename_WP, Avg, ScanLen, N1=0, N2=0) / AvgSamplesNumber
        
    return PE, TT, WP





def load_bin_acqs(Path: str, N_acqs: int, TT_and_PE: bool=True):
    '''
    Load a number of TT_Ascans and PE_Ascans from a binary file. The number of 
    scans is given by N_acqs. The TT and PE traces in the binary file should be
    interlaced and starting with a TT, meaning: TT_1, PE_1, TT_2, PE_2,...
    
    Two matrices are returned, one with all TTs and another one with all PEs.

    Parameters
    ----------
    Path : str
        File path.
    N_acqs : int
        Total number of acquisitions saved in the binary file.
    TT_and_PE : bool, optional
        If True, return two matrices with N_acqs columns each, one with TT
        scans and another one with PE scans. If False, return a signle matrix
        of N_acqs columns. The default is True.
    Returns
    -------
    TT_Ascans : ndarray
        2D matrix where each column represents one TT acquisition.
    PE_Ascans : ndarray
        2D matrix where each column represents one PE acquisition.

    Arnau, 10/01/2023
    '''
    with open(Path, 'rb') as f:
        data = np.fromfile(f)
    if TT_and_PE:
        data = np.split(data, 2*N_acqs)
        data = np.array(data).T
        TT_Ascans = data[:,::2]
        PE_Ascans = data[:,1::2]
        return TT_Ascans, PE_Ascans
    else:
        data = np.split(data, N_acqs)
        data = np.array(data).T
        return data

def saveDict2txt(Path: str, d: dict, mode: str='w', delimiter: str=','):
    '''
    Saves a dictionary to a text file where each row is formatted as:
        {key}{delimiter}{value}

    Supported modes are 'w' for write and 'a' for append.    

    Parameters
    ----------
    Path : str
        File path.
    d : dict
        Dictionary to save.
    mode : str, optional
        File opening mode. The default is 'w'.
    delimiter : str, optional
        Delimiter between key and value. The default is ','.

    Returns
    -------
    None.

    Arnau, 02/11/2022
    '''
    with open(Path, mode) as f:
        for key, value in d.items():
            f.write(f'{key}{delimiter}{value}\n')
            
def load_config(Path: str):
    '''
    Load an experiment's configuration file as a dictionary. The configuration
    file needs to have at least the following keys:
        Fs, Fs_Gencode_Generator, Gain_Ch1, Gain_Ch2, Attenuation_Ch1,
        Attenuation_Ch2, Excitation_voltage, Excitation_params, Smin1, Smin2,
        Smax1, Smax2, AvgSamplesNumber, Quantiz_Levels, Ts_acq, N_acqs.

    Parameters
    ----------
    Path : str
        File path.

    Returns
    -------
    d : dict
        Configuration dictionary.

    Arnau, 20/01/2023
    '''
    d = {}
    with open(Path, 'r') as f:
        data = f.read()
    data = data.split(sep='\n')[:-1]
    for s in data:
        key, value = s.split(',', maxsplit=1)
        d[key] = value
    
    
    def toBool(x):
        return x == 'True'
    
    def floatOrNone(x):
        return float(x) if x != 'None' else None
    
    types_dict = {'Fs': float,
                  'num_bits' : int,
                  'Fs_Gencode_Generator' : float,
                  'Gain_Ch1' : float,
                  'Gain_Ch2' : float,
                  'Attenuation_Ch1' : float,
                  'Attenuation_Ch2' : float,
                  'Attenuation_ChA' : float,
                  'Attenuation_ChB' : float,
                  'Attenuation_tx' : float,
                  'Attenuation_rx' : float,
                  'Excitation_voltage' : float,
                  'Excitation_params' : literal_eval,
                  'Smin1' : int,
                  'Smin2' : int,
                  'Smax1' : int,
                  'Smax2' : int,
                  'AvgSamplesNumber' : int,
                  'ScanLen' : int,
                  'Quantiz_Levels' : int,
                  'Ts_acq' : floatOrNone,
                  'N_acqs' : int,
                  'N_avg' : int,
                  'ID' : toBool,
                  'stripIterNo' : int,
                  'WP_temperature' : floatOrNone,
                  'Outside_temperature' : floatOrNone,
                  'Cw' : float,
                  'CW' : float,
                  'cw' : float,
                  'length' : floatOrNone,
                  'Length' : floatOrNone
                  }

    for k, v in types_dict.items():
        if k in d: d[k] = v(d[k])
    return d

def load_columnvectors_fromtxt(Path: str, delimiter: str=',', header: bool=True, dtype=float):
    '''
    Load data float from a text file. If header is True, then the first row of
    the file is interpreted as the header which is used as keys for the 
    returned dictionary. Otherwise, an MxN matrix is returned, where M is the 
    number of rows in the file and N is the number of columns.

    Parameters
    ----------
    Path : str
        File path.
    delimiter : str, optional
        Delimiter between values of a row. The default is ','.
    header : bool, optional
        If true, the first row of the file is interpreted as the header and a 
        dictionary is returned. The default is True.
    dtype : data-type, optional
        See numpy.loadtxt doc for more information. Default is float.
    
    Returns
    -------
    data : dict or ndarray
        data loaded from the text file.

    Arnau, 02/02/2023
    '''
    if header:
        d = {}
        headers = np.loadtxt(Path, dtype=str, delimiter=delimiter, max_rows=1)
        data = np.loadtxt(Path, delimiter=delimiter, skiprows=1, dtype=dtype)
        for i, h in enumerate(headers):
            d[h] = data[i] if data.ndim==1 else data[:,i]
        return d
    else:
        data = np.loadtxt(Path, delimiter=delimiter, dtype=dtype)
        return data

def load_all(Experiment_Path: str):
    '''
    Load all data of one experiment given the path of the experiment's folder.
    The file names are assumed to be:
        'config.txt'
        'results.txt'
        'WP.bin'
        'acqdata.bin'
        'temperature.txt'
        
    Parameters
    ----------
    Experiment_Path : str
        Absolute path of the experiment's folder.

    Returns
    -------
    config_dict : dict
        Configuration dictionary.
    results_dict : dict
        Results dictionary.
    temperature_dict : dict
        Temperature and Cw dictionary.
    WP_Ascan : ndarray
        Water Path data.
    TT_Ascans : ndarray
        2D matrix where each column represents one TT acquisition.
    PE_Ascans : ndarray
        2D matrix where each column represents one PE acquisition.

    Arnau, 08/11/2022
    '''
    Config_path = os.path.join(Experiment_Path, 'config.txt')
    Results_path = os.path.join(Experiment_Path, 'results.txt')
    WP_path = os.path.join(Experiment_Path, 'WP.bin')
    Acqdata_path = os.path.join(Experiment_Path, 'acqdata.bin')
    Temperature_path = os.path.join(Experiment_Path, 'temperature.txt')
    
    # Load config
    config_dict = load_config(Config_path)
    
    # Load results
    results_dict = load_columnvectors_fromtxt(Results_path)

    # Load temperature and Cw
    temperature_dict = load_columnvectors_fromtxt(Temperature_path)
    
    # Load WP
    with open(WP_path, 'rb') as f:
        WP_Ascan = np.fromfile(f)
    
    # Load acq data
    TT_Ascans, PE_Ascans = load_bin_acqs(Acqdata_path, config_dict['N_acqs'])
    
    return config_dict, results_dict, temperature_dict, WP_Ascan, TT_Ascans, PE_Ascans