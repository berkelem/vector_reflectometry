# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:39:15 2017

@author: Matthew Berkeley
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import os

class Datafile():

    
    def __init__(self, filename):
        '''Datafile with extension *.s2p from vector network analyzer.
    
        Attributes
        ----------
        
        .f : float
            Frequency (Hz) \n
        .s11r : float
            Real component of S11 \n
        .s11i : float
            Imaginary component of S11 \n
        .s21r : float
            Real component of S21 \n
        .s21i : float
            Imaginary component of S21 \n
        .s12r : float
            Real component of S12 \n
        .s12i : float
            Imaginary component of S12 \n
        .s22r : float
            Real component of S22 \n
        .s22i : float
            Imaginary component of S22 \n
        '''
        with open(filename, 'r') as f1:
            f,s11r,s11i,s21r,s21i,s12r,s12i,s22r,s22i=np.loadtxt(
                f1,skiprows=8,delimiter=' ',unpack=True)
        self.f = f
        self.s11r = s11r
        self.s11i = s11i
        self.s21r = s21r
        self.s21i = s21i
        self.s12r = s12r
        self.s12i = s12i
        self.s22r = s22r
        self.s22i = s22i


class Measurement():
    
    
    def __init__(self, thru_file, model_file=None):
        '''Measurement dataset.
        
        Attributes
        ----------
        
        .thru_file : str
            Filename for 'through' data.
        .bias : 1darray
            Measurement bias calculated from thru_file
        .model : bool
            True if model data to be included
        .model_file : str, optional
            Model filename
            
        
        Methods
        -------
        
        .calc_bias :
            Calculate the measurement bias from the thru_file.
            
        '''
        self.thru_file = thru_file
        self.bias = self.calc_bias()
        self.model = False
        if model_file is not None:
            self.model_file = model_file
            self.model = True
        
    def calc_bias(self, factor=4):
        '''Calculate |S21| from thru_file, smoothed along the top of noisy 
        data. The default smoothing factor smooths over 4 data points each way. 
        This can be tuned as needed.
        
        Returns
        -------
        bias : 1darray
            Each point is the maximum value of #factor points either side.
        
        '''
        thru = Datafile(self.thru_file)
        thru.f = np.array([x*10**-9 for x in thru.f])
        trans = np.sqrt(thru.s21r**2 + thru.s21i**2)
        bias = ceil_data(trans, factor=factor)
        bias = -20.*np.log10(bias)
        return bias
    
    def calc_response(self, file_list, distances, port):
        '''Calculate measured response of instrument and DUT at a given port.
        
        Parameters
        ----------
        
        file_list : list
            List of datafile names.
        distances : ndarray
            List of x-offset distances corresponding to each datafile.
        port : str
            One of ['S11', 'S21', 'S12', 'S22']
            
        Returns
        -------
        
        resp_inst : ndarray
            Response of the instrument.            
        resp_dut : ndarray
            Response of the Device Under Test (DUT)
        freq : ndarray
            Frequencies measured.
        '''
        
        S_real = []
        S_imag = []
        for i in xrange(len(file_list)):
            data = Datafile(file_list[i])
            if port == 'S11':
                S_real.append(data.s11r)
                S_imag.append(data.s11i)
            elif port == 'S21':
                S_real.append(data.s21r)
                S_imag.append(data.s21i)
            elif port == 'S12':
                S_real.append(data.s12r)
                S_imag.append(data.s12i)
            elif port == 'S22':
                S_real.append(data.s22r)
                S_imag.append(data.s22i)
            else:
                raise ValueError('Pick a port: S11, S21 or S22.')
            freq = data.f
        k = 2*np.pi*freq/const.c
        S_real = np.array(S_real)
        S_imag = np.array(S_imag)
        resp_inst = []
        resp_dut = []
        for j in xrange(len(k)):
            sumsin2kx = np.sum([np.sin(2*k[j]*distances[i]) 
                                for i in xrange(len(file_list))])
            sumcos2kx = np.sum([np.cos(2*k[j]*distances[i]) 
                                for i in xrange(len(file_list))])
            D = np.array([
                    np.sum([S_real[i][j] for i in xrange(len(file_list))]), 
                    np.sum([S_imag[i][j] for i in xrange(len(file_list))]),
                    np.sum([(S_imag[i][j]*np.sin(2*k[j]*distances[i]) 
                                + S_real[i][j]*np.cos(2*k[j]*distances[i])) 
                                for i in xrange(len(file_list))]), 
                    np.sum([(S_imag[i][j]*np.cos(2*k[j]*distances[i]) 
                                - S_real[i][j]*np.sin(2*k[j]*distances[i])) 
                                for i in xrange(len(file_list))])
                            ])
        
            A = np.array([
                    [len(file_list), 0.0, sumcos2kx, -sumsin2kx],
                    [0.0, len(file_list), sumsin2kx, sumcos2kx],
                    [sumcos2kx, sumsin2kx, len(file_list), 0.0],
                    [-sumsin2kx, sumcos2kx, 0.0, len(file_list)]
                     ])
            Ainv = np.linalg.inv(A)
            assert np.allclose(np.dot(A,Ainv), np.identity(4))
            gamma = np.dot(Ainv, D)
            if port == 'S11' or port == 'S22':
                inst = 20*np.log10(np.sqrt(gamma[0]**2 + gamma[1]**2))
                dut = 20*np.log10(np.sqrt(gamma[2]**2 + gamma[3]**2)) 
            elif port == 'S21' or port == 'S12':
                inst = 20*np.log10(np.sqrt(gamma[2]**2 + gamma[3]**2))
                dut = 20*np.log10(np.sqrt(gamma[0]**2 + gamma[1]**2)) 
            resp_inst.append(inst)
            resp_dut.append(dut)
        return np.array(resp_inst), np.array(resp_dut), freq

    def plot_reflectance(self, path, arg, distances, port='S11', scale='log', 
                         saveas='FSS_refl.png', band='Gband'):
        '''Plot reflectance.
        
        Parameters
        ----------
        
        path : str
            Path to directory containing data.
        arg : list
            List of identifying substrings contained in relevant filenames.
            e.g. ['th0', 'th5', 'th10', 'th15']
        distances : list
            List of distances at which measurements were taken.
        port : str, optional
            Default for reflectance is S11.
        scale : str, optional
            Default is 'log' scale, with reflectance in units of dB.
            Alternatively, choose 'linear'.
        saveas : str, optional
            Desired output filename. Default is 'FSS_refl.png'.
        '''
        
        freq_dict = {'Wband':[75,115], 'Gband':[140,220]}
        freq_range = freq_dict[band]
        
        file_list = make_filelist(path, arg)
        R_inst, R_dut, freq = self.calc_response(
                            file_list, np.array([0. - dist for dist in 
                                                distances]), 
                            port=port)
        if scale == 'linear':
            R_inst = np.array([10.0**(R_inst[i]/10.0) for i in 
                                xrange(len(R_inst))])
            R_dut = np.array([10.0**((R_dut[i]+self.bias)/10.0) for i in 
                                xrange(len(R_dut))])
            l1,l2 = plt.plot(freq*10**-9, R_inst, 'r', freq*10**-9, R_dut, 'b')
        elif scale == 'log':
            l1,l2 = plt.plot(freq*10**-9, R_inst, 'r', freq*10**-9, 
                             R_dut + self.bias, 'b')
        else:
            raise ValueError('Pick a scale: either "linear" or "log".')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Reflectance (dB)')
        if scale == 'linear':
            plt.ylabel('Reflectance')
        plt.legend((l1,l2), ('Inst', 'DUT'))
        plt.xlim([freq_range[0]-10,freq_range[1]+10])
        plt.title(arg)
        plt.savefig(saveas)
        plt.close()
        return
    
    def plot_transmittance(self, path, arg, distances, port='S21', scale='linear', saveas='FSS_trans.png', band='Gband'):
        '''Plot transmittance.
        
        Parameters
        ----------
        
        path : str
            Path to directory containing data.
        arg : list
            List of identifying substrings contained in relevant filenames.
            e.g. ['th0', 'th5', 'th10', 'th15']
        distances : list
            List of distances at which measurements were taken.
        port : str, optional
            Default for reflectance is S21.
        scale : str, optional
            Default is 'linear' scale.
            Alternatively, choose 'log'.
        saveas : str, optional
            Desired output filename. Default is 'FSS_trans.png'.
        '''
        freq_dict = {'Wband':[75,115], 'Gband':[140,220]}
        freq_range = freq_dict[band]
        
        file_list = make_filelist(path, arg)
        T_inst, T_dut, freq = self.calc_response(
                            file_list, np.array([0. - dist for dist in distances]), 
                            port=port)
        if scale == 'linear':
            T_inst = np.array([10.0**(T_inst[i]/10.0) for i in xrange(len(T_inst))])
            T_dut = np.array([10.0**((T_dut[i]+self.bias[i])/10.0) for i in xrange(len(T_dut))])
        freq = freq*10**-9
        if self.model:
            with open(model_file, 'r') as f1:
                f, deg0, deg5, deg10, deg15, deg20, deg25, deg30, deg35, deg40 = np.genfromtxt(f1, dtype=None, skip_header=1, usecols=range(10),  unpack=True)
            if arg == 'th0' or arg == 'Th0':
                model = deg0
            elif arg == 'th5' or arg == 'Th5':
                model = deg5
            elif arg == 'th10' or arg == 'Th10':
                model = deg10
            elif arg == 'th15' or arg == 'Th15':
                model = deg15
            elif arg == 'th20' or arg == 'Th20':
                model = deg20
            elif arg == 'th25' or arg == 'Th25':
                model = deg25
            elif arg == 'th30' or arg == 'Th30':
                model = deg30
            elif arg == 'th35' or arg == 'Th35':
                model = deg35
            elif arg == 'th40' or arg == 'Th40':
                model = deg40
            else:
                raise ValueError('Argument does not match model options.')
            if scale == 'log':
                model = 20*np.log10(model)
                T_dut_corr = T_dut + self.bias
                l1,l2,l3 = plt.plot(freq[(freq>=freq_range[0])&(freq<=freq_range[1])], T_inst[(freq>=freq_range[0])&(freq<=freq_range[1])], 'r', freq[(freq>=freq_range[0])&(freq<=freq_range[1])], T_dut_corr[(freq>=freq_range[0])&(freq<=freq_range[1])], 'b', f, model, 'g')
            elif scale == 'linear':            
                l1,l2,l3 = plt.plot(freq[(freq>=freq_range[0])&(freq<=freq_range[1])], T_inst[(freq>=freq_range[0])&(freq<=freq_range[1])], 'r', freq[(freq>=freq_range[0])&(freq<=freq_range[1])], ceil_data(T_dut, factor=3)[(freq>=freq_range[0])&(freq<=freq_range[1])], 'b', f, model, 'g')
            else:
                raise ValueError('Pick a scale: either "linear" or "log".')
            plt.legend((l1,l2,l3), ('Inst', 'DUT','Model'), loc='lower left')
        else:
            if scale == 'log':
                l1,l2 = plt.plot(freq*10**-9, T_inst, 'r', freq*10**-9, T_dut + self.bias, 'b')
            elif scale == 'linear':
                l1,l2 = plt.plot(freq*10**-9, T_inst, 'r', freq*10**-9, T_dut, 'b')
            else:
                raise ValueError('Pick a scale: either "linear" or "log".')
            plt.legend((l1,l2), ('Inst', 'DUT'))
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Transmittance (dB)')
        if scale == 'linear':
            plt.ylabel('Transmittance')
            #plt.ylim([0,1.0])
        plt.xlim([freq_range[0]-10,freq_range[1]+10])
        plt.title(arg)
        plt.savefig(saveas)
        plt.close()
        data_to_write = np.array([freq, T_inst, T_dut, self.bias])
        with open(path+'plots/FSS_trans_'+str(arg)+'.dat', 'w') as f:
            f.write('Frequency(GHz) T_inst T_dut Bias \n')
            np.savetxt(f, data_to_write.T, fmt = '%f')
        return
        
def smooth_data(data, factor=5):
    '''Smooth data taking averages over #factor data points either side.
    Default is factor=5.
    '''
    smooth_data = np.zeros_like(np.array(data))
    for i in xrange(len(data)):
        smooth_data[i] = (np.sum(
            data[max(i-factor,0):min(i+factor, len(data)-1)])
            /(min(i+factor, len(data)-1)-max(i-factor,0))
            )
    return smooth_data

def ceil_data(data, factor=1):
    '''Smooth data along the ceiling value of a noisy curve.'''
    ceil_data = np.zeros_like(np.array(data))
    for i in xrange(len(data)):
        ceil_data[i] = max(data[max(i-factor,0):min(i+factor, len(data)-1)])
    return ceil_data

def make_filelist(path, arg):
    '''Create a list of files from the desired path and identifying arg.'''
    file_list = []
    for files in os.listdir(path):
        if arg in files and '_1.' not in files:
            file_list.append(path+files)
    return file_list



if __name__ == '__main__':
    model_file = '3mm_ind.txt'
    thru_file = '../../Network\ Analyzer\ Projects/Mesh\ filter\ experiment/020718/thru_0.s2p'
    m = Measurement(thru_file, model_file)
    
    path = './020718/'
    args = ['Th0', 'Th30','Th40']
    distances = [0.0254*d for d in [-0.0130, -0.0066, 0., 0.0066, 0.0130]]
    
    for arg in args:
        m.plot_reflectance(path, arg, distances, saveas=path+'plots/FSS_refl_'+arg+'.png', band='Wband')
        m.plot_transmittance(path, arg, [-x for x in distances], saveas=path+'plots/FSS_trans_'+arg+'.png', port='S21', band='Gband')
