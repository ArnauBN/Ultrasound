# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 08:46:32 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import serial, time

#%%
class Scanner():
    def __init__(self, port='COM4', baudrate=19200, timeout=0.1):
        
        # Open Serial communication
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.port = port

        # Steps in mm
        self.uStepX = 0.01
        self.uStepY = 0.01
        self.uStepZ = 0.005
        self.uStepR = 1.8 # (200steps=360deg) --- the other value that doesnt work is 0.0281

        # Current coordinates
        self.getCoords()
        
        # Enable all by default
        self.enableAll()

        # Set default directions to: '+', '+', '-', '+'
        self.setDirections()

        # Set default speedtypes to 'rectangular'
        self.setSpeedtypes()
    
        # Set default speeds (mm/s): 100, 100, 100, 100
        self.setSpeeds()
        

    def getuSteps(self, axis):
        '''
        Returns the uStep of the corresponding axis.

        Parameters
        ----------
        axis : str or int
            Available axis are:
                'X' or 0
                'Y' or 1
                'Z' or 2
                'R' or 3

        Returns
        -------
        uStep : float
            The corresponding uStep.

        Arnau, 01/02/2023
        '''
        if axis=='X' or axis==0:
            return self.uStepX
        elif axis=='Y' or axis==1:
            return self.uStepY
        elif axis=='Z' or axis==2:
            return self.uStepZ
        elif axis=='R' or axis==3:
            return self.uStepR
    
    def uSteps2value(self, axis, uSteps):
        return float(uSteps) * self.getuSteps(axis)
    
    def value2uSteps(self, axis, value):
        return int(value / self.getuSteps(axis))
    
    
    # ========= WRITE COMMAND =========
    def write(self, cmd):
        try:
            if self.ser.isOpen() == False:
                self.ser.open()
                print(f'Scanner {self.port}> Port opened')
        except Exception as e:
            print(f'Scanner {self.port}> error communicating...: {e}')
            self.ser.close()

        if self.ser.isOpen():
            try:
                self.ser.flushInput()
                self.ser.flushOutput()
                self.ser.write((cmd + '\r').encode('utf-8'))
                time.sleep(0.1)
                response = self.ser.read(10)
                if cmd[:2] in ['SM', 'SL', 'SC', 'SD', 'SN', 'SA', 'SG']:
                    while response == b'':
                        time.sleep(0.2)
                        response = self.ser.read(10)
                return response
            except KeyboardInterrupt:
                self.ser.write('SSF\r'.encode('utf-8'))
                print('Scanner successfully stopped.')
                raise
            except Exception as e1:
                self.ser.close()
                print(f'Scanner {self.port}> error communicating...: {e1}')
                print(f'Scanner {self.port}> Port closed')


    # ========== READ COORDINATES ==========
    @property
    def X(self):
        x = self.write('SCX')
        X = float(x[3:]) * self.uStepX
        self._X = X
        return X

    @property
    def Y(self):
        y = self.write('SCY')
        Y = float(y[3:]) * self.uStepY
        self._Y = Y
        return Y
    
    @property
    def Z(self):
        z = self.write('SCZ')
        Z = float(z[3:]) * self.uStepZ
        self._Z = Z
        return Z
    
    @property
    def R(self):
        r = self.write('SCR')
        R = float(r[3:]) * self.uStepR
        self._R = R
        return R

    def getCoords(self):
        return self.X, self.Y, self.Z, self.R

    def getAxis(self, axis):
        if axis.upper() == 'X':
            return self.X
        elif axis.upper() == 'Y':
            return self.Y
        elif axis.upper() == 'Z':
            return self.Z
        elif axis.upper() == 'R':
            return self.R
        

    # ======= SET CURRENT =======
    @X.setter
    def X(self, value):
        steps = self.value2uSteps('X', value)
        if self.write(f'SAX{steps}')[:2] != b'OK':
            print(f'Could not set current absolute X axis position as {value} mm.')
            return
        self._X = value

    @Y.setter
    def Y(self, value):
        steps = self.value2uSteps('Y', value)
        if self.write(f'SAY{steps}')[:2] != b'OK':
            print(f'Could not set current absolute Y axis position as {value} mm.')
            return
        self._Y = value
    
    @Z.setter
    def Z(self, value):
        steps = self.value2uSteps('Z', value)
        if self.write(f'SAZ{steps}')[:2] != b'OK':
            print(f'Could not set current absolute Z axis position as {value} mm.')
            return
        self._Z = value
    
    @R.setter
    def R(self, value):
        steps = self.value2uSteps('R', value)
        if self.write(f'SAR{steps}')[:2] != b'OK':
            print(f'Could not set current absolute R axis position as {value} deg.')
            return
        self._R = value
    
    def setCurrent(self, Xvalue, Yvalue, Zvalue, Rvalue):
        self.X = Xvalue
        self.Y = Yvalue
        self.Z = Zvalue
        self.R = Rvalue
    
    def setZero(self):
        self.setCurrent(0, 0, 0, 0)

    def setAxis(self, axis, value):
        if axis.upper() == 'X':
            self.X = value
        elif axis.upper() == 'Y':
            self.Y = value
        elif axis.upper() == 'Z':
            self.Z = value
        elif axis.upper() == 'R':
            self.R = value


    # ======= MOVEMENT =======
    def moveX(self, value):
        steps = self.value2uSteps('X', value)
        if self.write(f'SMX{steps}')[:2] != b'OK':
            print(f'Could not move to X axis absolute coordinate {value} mm.')
            return
        self._X = value

    def moveY(self, value):
        steps = self.value2uSteps('Y', value)
        if self.write(f'SMY{steps}')[:2] != b'OK':
            print(f'Could not move to Y axis absolute coordinate {value} mm.')
            return
        self._Y = value
    
    def moveZ(self, value):
        steps = self.value2uSteps('Z', value)
        if self.write(f'SMZ{steps}')[:2] != b'OK':
            print(f'Could not move to Z axis absolute coordinate {value} mm.')
            return
        self._Z = value
    
    def moveR(self, value):
        steps = self.value2uSteps('R', value)
        if self.write(f'SMR{steps}')[:2] != b'OK':
            print(f'Could not move to R axis absolute coordinate {value} deg.')
            return
        self._R = value
    
    def move(self, Xvalue, Yvalue, Zvalue, Rvalue):
        self.moveX(Xvalue)
        self.moveY(Yvalue)
        self.moveZ(Zvalue)
        self.moveR(Rvalue)
    
    def goHome(self):
        self.move(0, 0, 0, 0)

    def moveAxis(self, axis, value):
        if axis.upper() == 'X':
            self.moveX(value)
        elif axis.upper() == 'Y':
            self.moveY(value)
        elif axis.upper() == 'Z':
            self.moveZ(value)
        elif axis.upper() == 'R':
            self.moveR(value)
    

    # ======= DIFFERENTIAL MOVEMENT =======
    def diffMoveX(self, value):
        steps = self.value2uSteps('X', value)
        if self.write(f'SDX{steps}')[:2] != b'OK':
            print(f'Could not move X axis relative coordinate by {value} mm.')
            return
        self._X = value
    
    def diffMoveY(self, value):
        steps = self.value2uSteps('Y', value)
        if self.write(f'SDY{steps}')[:2] != b'OK':
            print(f'Could not move Y axis relative coordinate by {value} mm.')
            return
        self._Y = value
    
    def diffMoveZ(self, value):
        steps = self.value2uSteps('Z', value)
        if self.write(f'SDZ{steps}')[:2] != b'OK':
            print(f'Could not move Z axis relative coordinate by {value} mm.')
            return
        self._Z = value
    
    def diffMoveR(self, value):
        steps = self.value2uSteps('R', value)
        if self.write(f'SDR{steps}')[:2] != b'OK':
            print(f'Could not move R axis relative coordinate by {value} deg.')
            return
        self._R = value
    
    def diffMove(self, Xvalue, Yvalue, Zvalue, Rvalue):
        self.diffMoveX(Xvalue)
        self.diffMoveY(Yvalue)
        self.diffMoveZ(Zvalue)
        self.diffMoveR(Rvalue)
    
    def diffMoveAxis(self, axis, value):
        if axis.upper() == 'X':
            self.diffMoveX(value)
        elif axis.upper() == 'Y':
            self.diffMoveY(value)
        elif axis.upper() == 'Z':
            self.diffMoveZ(value)
        elif axis.upper() == 'R':
            self.diffMoveR(value)
    
    
    # ======= UNLIMITED DIFFERENTIAL MOVEMENT =======
    def unlimitedDiffMoveX(self, value):
        steps = self.value2uSteps('X', value)
        if self.write(f'SNX{steps}')[:2] != b'OK':
            print(f'Could not move X axis relative coordinate by {value} mm.')
            return
        self._X = value
    
    def unlimitedDiffMoveY(self, value):
        steps = self.value2uSteps('Y', value)
        if self.write(f'SNY{steps}')[:2] != b'OK':
            print(f'Could not move Y axis relative coordinate by {value} mm.')
            return
        self._Y = value
    
    def unlimitedDiffMoveZ(self, value):
        steps = self.value2uSteps('Z', value)
        if self.write(f'SNZ{steps}')[:2] != b'OK':
            print(f'Could not move Z axis relative coordinate by {value} mm.')
            return
        self._Z = value
    
    def unlimitedDiffMoveR(self, value):
        steps = self.value2uSteps('R', value)
        if self.write(f'SNR{steps}')[:2] != b'OK':
            print(f'Could not move R axis relative coordinate by {value} deg.')
            return
        self._R = value
    
    def unlimitedDiffMove(self, Xvalue, Yvalue, Zvalue, Rvalue):
        self.unlimitedDiffMoveX(Xvalue)
        self.unlimitedDiffMoveY(Yvalue)
        self.unlimitedDiffMoveZ(Zvalue)
        self.unlimitedDiffMoveR(Rvalue)

    def unlimitedDiffMoveAxis(self, axis, value):
        if axis.upper() == 'X':
            self.unlimitedDiffMoveX(value)
        elif axis.upper() == 'Y':
            self.unlimitedDiffMoveY(value)
        elif axis.upper() == 'Z':
            self.unlimitedDiffMoveZ(value)
        elif axis.upper() == 'R':
            self.unlimitedDiffMoveR(value)
    

    # ======== GET LIMITS ========
    @property
    def XLimit(self):
        response = self.write('SGX')
        Xlim = self.uSteps2value('X', float(response[3:]))
        return Xlim

    @property
    def YLimit(self):
        response = self.write('SGY')
        Ylim = self.uSteps2value('Y', float(response[3:]))
        return Ylim

    @property
    def ZLimit(self):
        response = self.write('SGZ')
        Zlim = self.uSteps2value('Z', float(response[3:]))
        return Zlim

    @property
    def RLimit(self):
        response = self.write('SGR')
        Rlim = self.uSteps2value('R', float(response[3:]))
        return Rlim
    
    def getLimits(self):
        return self.XLimit, self.YLimit, self.ZLimit, self.RLimit

    def getAxisLimit(self, axis):
        if axis.upper() == 'X':
            return self.XLimit
        elif axis.upper() == 'Y':
            return self.YLimit
        elif axis.upper() == 'Z':
            return self.ZLimit
        elif axis.upper() == 'R':
            return self.RLimit
    

    # ======== SET LIMITS ========
    @XLimit.setter
    def XLimit(self, limit):
        l = self.value2uSteps('X', limit)
        if self.write(f'SLX{l}')[:2] != b'OK':
            print(f'Could not set X axis max coordinate limit to {limit} mm.')

    @YLimit.setter
    def YLimit(self, limit):
        l = self.value2uSteps('Y', limit)
        if self.write(f'SLY{l}')[:2] != b'OK':
            print(f'Could not set Y axis max coordinate limit to {limit} mm.')

    @ZLimit.setter
    def ZLimit(self, limit):
        l = self.value2uSteps('Z', limit)
        if self.write(f'SLZ{l}')[:2] != b'OK':
            print(f'Could not set Z axis max coordinate limit to {limit} mm.')
    
    @RLimit.setter    
    def RLimit(self, limit):
        l = self.value2uSteps('R', limit)
        if self.write(f'SLR{l}')[:2] != b'OK':
            print(f'Could not set R axis max coordinate limit to {limit} deg.')

    def setLimits(self, Xlim, Ylim, Zlim, Rlim):
        self.XLimit = Xlim
        self.YLimit = Ylim
        self.ZLimit = Zlim
        self.RLimit = Rlim

    def setAxisLimit(self, axis, value):
        if axis.upper() == 'X':
            self.XLimit = value
        elif axis.upper() == 'Y':
            self.YLimit = value
        elif axis.upper() == 'Z':
            self.ZLimit = value
        elif axis.upper() == 'R':
            self.RLimit = value
    

    # ========= SWAP DIRECTION =========
    @property
    def Xdirection(self):
        return self._Xdirection

    @property
    def Ydirection(self):
        return self._Ydirection

    @property
    def Zdirection(self):
        return self._Zdirection

    @property
    def Rdirection(self):
        return self._Rdirection
    
    def getAxisDirection(self, axis):
        if axis.upper() == 'X':
            return self.Xdirection
        elif axis.upper() == 'Y':
            return self.Ydirection
        elif axis.upper() == 'Z':
            return self.Zdirection
        elif axis.upper() == 'R':
            return self.Rdirection
    
    @Xdirection.setter
    def Xdirection(self, direction):
        self.write(f'SWX{direction}')
        self._Xdirection = direction

    @Ydirection.setter
    def Ydirection(self, direction):
        self.write(f'SWY{direction}')
        self._Ydirection = direction
    
    @Zdirection.setter
    def Zdirection(self, direction):
        self.write(f'SWZ{direction}')
        self._Zdirection = direction
    
    @Rdirection.setter
    def Rdirection(self, direction):
        self.write(f'SWR{direction}')
        self._Rdirection = direction

    def setDirections(self, Xdirection='+', Ydirection='+', Zdirection='-', Rdirection='+'):
        self.Xdirection = Xdirection
        self.Ydirection = Ydirection
        self.Zdirection = Zdirection
        self.Rdirection = Rdirection
        
    def setAxisDirection(self, axis, value):
        if axis.upper() == 'X':
            self.Xdirection = value
        elif axis.upper() == 'Y':
            self.Ydirection = value
        elif axis.upper() == 'Z':
            self.Zdirection = value
        elif axis.upper() == 'R':
            self.Rdirection = value
    

    # ========= SET SPEEDTYPE =========
    @property
    def Xspeedtype(self):
        return self._Xspeedtype
 
    @property
    def Yspeedtype(self):
        return self._Yspeedtype
    
    @property
    def Zspeedtype(self):
        return self._Zspeedtype
    
    @property
    def Rspeedtype(self):
        return self._Rspeedtype
    
    def getAxisSpeedtype(self, axis):
        if axis.upper() == 'X':
            return self.Xspeedtype
        elif axis.upper() == 'Y':
            return self.Yspeedtype
        elif axis.upper() == 'Z':
            return self.Zspeedtype
        elif axis.upper() == 'R':
            return self.Rspeedtype
    
    @Xspeedtype.setter
    def Xspeedtype(self, speedtype):
        code = _parseSpeedtype(speedtype)
        self.write(f'STX{code}')
        self._Xspeedtype = code

    @Yspeedtype.setter
    def Yspeedtype(self, speedtype):
        code = _parseSpeedtype(speedtype)
        self.write(f'STY{code}')
        self._Yspeedtype = code

    @Zspeedtype.setter
    def Zspeedtype(self, speedtype):
        code = _parseSpeedtype(speedtype)
        self.write(f'STZ{code}')
        self._Zspeedtype = code

    @Rspeedtype.setter
    def Rspeedtype(self, speedtype):
        code = _parseSpeedtype(speedtype)
        self.write(f'STR{code}')
        self._Rspeedtype = code        
    
    def setSpeedtypes(self, Xspeedtype=0, Yspeedtype=0, Zspeedtype=0, Rspeedtype=0):
        self.Xspeedtype = Xspeedtype
        self.Yspeedtype = Yspeedtype
        self.Zspeedtype = Zspeedtype
        self.Rspeedtype = Rspeedtype

    def setAxisSpeedtype(self, axis, value):
        if axis.upper() == 'X':
            self.Xspeedtype = value
        elif axis.upper() == 'Y':
            self.Yspeedtype = value
        elif axis.upper() == 'Z':
            self.Zspeedtype = value
        elif axis.upper() == 'R':
            self.Rspeedtype = value
    

    # ======== GET/SET SPEED ========
    @property
    def Xspeed(self):
        return self._Xspeed

    @property
    def Yspeed(self):
        return self._Yspeed

    @property
    def Zspeed(self):
        return self._Zspeed

    @property
    def Rspeed(self):
        return self._Rspeed
    
    def getAxisSpeed(self, axis):
        if axis.upper() == 'X':
            return self.Xspeed
        elif axis.upper() == 'Y':
            return self.Yspeed
        elif axis.upper() == 'Z':
            return self.Zspeed
        elif axis.upper() == 'R':
            return self.Rspeed
    
    @Xspeed.setter
    def Xspeed(self, value):
        if value < 1 or value > 65536:
            print('Speed out of range. Ignoring...')
            return
        self.write(f'SPX{value}')
        self._Xspeed = value

    @Yspeed.setter
    def Yspeed(self, value):
        if value < 1 or value > 65536:
            print('Speed out of range. Ignoring...')
            return
        self.write(f'SPY{value}')
        self._Yspeed = value
        
    @Zspeed.setter
    def Zspeed(self, value):
        if value < 1 or value > 65536:
            print('Speed out of range. Ignoring...')
            return
        self.write(f'SPZ{value}')
        self._Zspeed = value
    
    @Rspeed.setter
    def Rspeed(self, value):
        if value < 1 or value > 65536:
            print('Speed out of range. Ignoring...')
            return
        self.write(f'SPR{value}')
        self._Rspeed = value

    def setSpeeds(self, Xvalue=100, Yvalue=100, Zvalue=100, Rvalue=100):
        self.Xspeed = Xvalue
        self.Yspeed = Yvalue
        self.Zspeed = Zvalue
        self.Rspeed = Rvalue

    def setAxisSpeed(self, axis, value):
        if axis.upper() == 'X':
            self.Xspeed = value
        elif axis.upper() == 'Y':
            self.Yspeed = value
        elif axis.upper() == 'Z':
            self.Zspeed = value
        elif axis.upper() == 'R':
            self.Rspeed = value
    

    # ======== GET/SET RAMPING SPEED ========
    @property
    def XRampingSpeed(self):
        return self._XRampingSpeed

    @property
    def YRampingSpeed(self):
        return self._YRampingSpeed

    @property
    def ZRampingSpeed(self):
        return self._ZRampingSpeed

    @property
    def RRampingSpeed(self):
        return self._RRampingSpeed    
    
    def getAxisRampingSpeed(self, axis):
        if axis.upper() == 'X':
            return self.XRampingSpeed
        elif axis.upper() == 'Y':
            return self.YRampingSpeed
        elif axis.upper() == 'Z':
            return self.ZRampingSpeed
        elif axis.upper() == 'R':
            return self.RRampingSpeed
    
    @XRampingSpeed.setter
    def XRampingSpeed(self, microseconds=50):
        if self.Xspeedtype not in [1,3]:
            print(f'Speedtype must be Gaussian (1) or Triangle (3), but {self.Xspeedtype} was found. Ignoring...')
            return
        self.write(f'SOX{microseconds}')
        self._XRampingSpeed = microseconds
    
    @YRampingSpeed.setter
    def YRampingSpeed(self, microseconds=50):
        if self.Yspeedtype not in [1,3]:
            print(f'Speedtype must be Gaussian (1) or Triangle (3), but {self.Yspeedtype} was found. Ignoring...')
            return
        self.write(f'SOY{microseconds}')
        self._YRampingSpeed = microseconds
    
    @ZRampingSpeed.setter
    def ZRampingSpeed(self, microseconds=50):
        if self.Zspeedtype not in [1,3]:
            print(f'Speedtype must be Gaussian (1) or Triangle (3), but {self.Zspeedtype} was found. Ignoring...')
            return
        self.write(f'SOZ{microseconds}')
        self._ZRampingSpeed = microseconds
    
    @RRampingSpeed.setter
    def RRampingSpeed(self, microseconds=50):
        if self.Rspeedtype not in [1,3]:
            print(f'Speedtype must be Gaussian (1) or Triangle (3), but {self.Rspeedtype} was found. Ignoring...')
            return
        self.write(f'SOR{microseconds}')
        self._RRampingSpeed = microseconds
    
    def setRampingSpeeds(self, Xmicroseconds=50, Ymicroseconds=50, Zmicroseconds=50, Rmicroseconds=50):
        self.XRampingSpeed = Xmicroseconds
        self.YRampingSpeed = Ymicroseconds
        self.ZRampingSpeed = Zmicroseconds
        self.RRampingSpeed = Rmicroseconds
    
    def setAxisRampingSpeed(self, axis, value):
        if axis.upper() == 'X':
            self.XRampingSpeed = value
        elif axis.upper() == 'Y':
            self.YRampingSpeed = value
        elif axis.upper() == 'Z':
            self.ZRampingSpeed = value
        elif axis.upper() == 'R':
            self.RRampingSpeed = value
    

    # ======== GET/SET RANDOM SPEED PARAMETER ========
    @property
    def XRandomSpeed(self):
        return self._XRandomSpeed
    
    @property
    def YRandomSpeed(self):
        return self._YRandomSpeed
    
    @property
    def ZRandomSpeed(self):
        return self._ZRandomSpeed
    
    @property
    def RRandomSpeed(self):
        return self._RRandomSpeed

    def getAxisRandomSpeed(self, axis):
        if axis.upper() == 'X':
            return self.XRandomSpeed
        elif axis.upper() == 'Y':
            return self.YRandomSpeed
        elif axis.upper() == 'Z':
            return self.ZRandomSpeed
        elif axis.upper() == 'R':
            return self.RRandomSpeed

    @XRandomSpeed.setter
    def XRandomSpeed(self, value):
        if self._Xspeedtype != 2:
            print(f'Speedtype must be Random (2), but {self.Xspeedtype} was found. Ignoring...')
            return
        if value < 1 or value > 65536:
            print('Speed out of range. Ignoring...')
            return
        self.write('SRX{value}')
        self._XRandomSpeed = value
    
    @YRandomSpeed.setter
    def YRandomSpeed(self, value):
        if self._Yspeedtype != 2:
            print(f'Speedtype must be Random (2), but {self.Yspeedtype} was found. Ignoring...')
            return
        if value < 1 or value > 65536:
            print('Speed out of range. Ignoring...')
            return
        self.write('SRY{value}')
        self._YRandomSpeed = value
    
    @ZRandomSpeed.setter
    def ZRandomSpeed(self, value):
        if self._Zspeedtype != 2:
            print(f'Speedtype must be Random (2), but {self.Zspeedtype} was found. Ignoring...')
            return
        if value < 1 or value > 65536:
            print('Speed out of range. Ignoring...')
            return
        self.write('SRZ{value}')
        self._ZRandomSpeed = value
    
    @RRandomSpeed.setter
    def RRandomSpeed(self, value):
        if self._Rspeedtype != 2:
            print(f'Speedtype must be Random (2), but {self.Rspeedtype} was found. Ignoring...')
            return
        if value < 1 or value > 65536:
            print('Speed out of range. Ignoring...')
            return
        self.write('SRR{value}')
        self._RRandomSpeed = value

    def setRandomSpeeds(self, Xvalue, Yvalue, Zvalue, Rvalue):
        self.XRandomSpeed = Xvalue
        self.YRandomSpeed = Yvalue
        self.ZRandomSpeed = Zvalue
        self.RRandomSpeed = Rvalue

    def setAxisRandomSpeed(self, axis, value):
        if axis.upper() == 'X':
            self.XRandomSpeed = value
        elif axis.upper() == 'Y':
            self.YRandomSpeed = value
        elif axis.upper() == 'Z':
            self.ZRandomSpeed = value
        elif axis.upper() == 'R':
            self.RRandomSpeed = value
    

    # ======== OPEN, CLOSE, STOP, ENABLE ========
    def close(self):
        self.ser.close()

    def open(self):
        try:
            if not self.ser.isOpen():
                self.ser.open()
                print(f'Scanner {self.port}> Port opened')
            else:
                print(f'Scanner {self.port}> Port already open')
        except Exception as e:
            print(f"Can't open serial port: {e}")

    def stop(self):
        self.write('SSF')
    
    def enableX(self, Enable=True):
        c = '+' if Enable else '-'
        self.write(f'SEX{c}')
        self.__Xenable = Enable

    def enableY(self, Enable=True):
        c = '+' if Enable else '-'
        self.write(f'SEY{c}')
        self.__Yenable = Enable
    
    def enableZ(self, Enable=True):
        c = '+' if Enable else '-'
        self.write(f'SEZ{c}')
        self.__Zenable = Enable
    
    def enableR(self, Enable=True):
        c = '+' if Enable else '-'
        self.write(f'SER{c}')
        self.__Renable = Enable
    
    def enable(self, XEnable=True, YEnable=True, ZEnable=True, REnable=True):
        self.enableX(XEnable)
        self.enableY(YEnable)
        self.enableZ(ZEnable)
        self.enableR(REnable)
    
    def enableAxis(self, axis, Enable=True):
        if axis.upper() == 'X':
            self.enableX(Enable)
        elif axis.upper() == 'Y':
            self.enableY(Enable)
        elif axis.upper() == 'Z':
            self.enableZ(Enable)
        elif axis.upper() == 'R':
            self.enableR(Enable)
    
    def enableAll(self):
        self.enable(True, True, True, True)

    def disableAll(self):
        self.enable(False, False, False, False)
    
    def isXEnabled(self):
        return self.__Xenable
    
    def isYEnabled(self):
        return self.__Yenable
    
    def isZEnabled(self):
        return self.__Zenable
    
    def isREnabled(self):
        return self.__Renable
   
    def isAxisEnabled(self, axis):
        if axis.upper() == 'X':
            return self.__Xenable
        elif axis.upper() == 'Y':
            return self.__Yenable
        elif axis.upper() == 'Z':
            return self.__Zenable
        elif axis.upper() == 'R':
            return self.__Renable


    # ======== EXPERIMENTS ========
    def checkside(self, SeDaq, Smin2, Smax2, axis, step, Max0, Maxtolerance):
        '''
        Moves the scanner over the specified axis until the echo is no longer
        received, then returns the distance travelled and goes back to the 
        original position.

        Parameters
        ----------
        SeDaq : SeDaq object
            Handler for the acquisition of the echo. The method 
            SeDaq.GetAscan_Ch2(Smin2, Smax2) is called.
        Smin2 : int
            First sample to capture. See SeDaq.GetAscan_Ch2(Smin2, Smax2).
        Smax2 : int
            Last sample to capture. See SeDaq.GetAscan_Ch2(Smin2, Smax2).
        axis : str
            Axis to move scanner on. Available axis are:
                'X'
                'Y'
                'Z'
                'R'
        step : float
            Precision in millimeters of the scanner.
        Max0 : float
            Maximum value of the original echo. This is used to determine
            if the echo is lost.
        Maxtolerance : float
            The echo is lost if:
                abs(Max0 - Max) > Maxtolerance.

        Returns
        -------
        x : int
            The total distance moved until the echo was lost (in millimeters).

        Arnau, 06/02/2023
        '''
        x = 0
        while True:
            self.diffMoveAxis(axis, step); x += step
            PE = SeDaq.GetAscan_Ch2(Smin2, Smax2)
            Max = np.max(np.abs(PE))
            if abs(Max - Max0) > Maxtolerance:
                self.diffMoveAxis(axis, step); x += step
                PE = SeDaq.GetAscan_Ch2(Smin2, Smax2)
                Max = np.max(np.abs(PE))
                if abs(Max - Max0) > Maxtolerance: # check twice just in case
                    x -= step
                    self.diffMoveAxis(axis, -step)
                    break
        self.diffMoveAxis(axis, -x)
        return x
    
    



def _parseSpeedtype(speedtype):
    '''
    Transform a string-type speed type to an ineger code (0,1,2 or 3).
    
    Available speed types are:
        'rectangular' -> 0
        'gaussian'    -> 1
        'random'      -> 2
        'triangle'    -> 3


    Parameters
    ----------
    speedtype : str or int
        Available speed types are:
            'rectangular' -> 0
            'gaussian'    -> 1
            'random'      -> 2
            'triangle'    -> 3

    Returns
    -------
    code : int
        Corresponding code.

    Arnau 01/02/2023
    '''
    allowed_speedtypes_str = ['rectangular', 'gaussian', 'random', 'triangle']
    if isinstance(speedtype, str):
        speedtype_lower = speedtype.lower()
        if speedtype_lower in allowed_speedtypes_str:
            code = allowed_speedtypes_str.index(speedtype_lower)
    elif speedtype in [0,1,2,3]:
        code = speedtype
    else:
        print("Wrong speed type. Using default ('rectangular').")
        return 0
    return code     



def makeScanPattern(pattern: str, steps: list, ends: list) -> list:
    '''
    Returns a list of strings. These strings are comprised of one letter 
    indicating the axis ('X', 'Y', 'Z' or 'R') and one number (can have a 
    decimal point and can be negative), in that order. Examples:
        'X1.5'
        'Y-40'
        'R90'
    The number is always the specified step.
    
    Available patterns are:
        'line on X'
        'line on Y'
        'line on Z'
        'line+turn on X'
        'line+turn on Y'
        'line+turn on Z'
        'zigzag XY'
        'zigzag XZ'
        'zigzag YX'
        'zigzag YZ'
        'zigzag ZX'
        'zigzag ZY'
        'zigzag+turn XY'
        'zigzag+turn XZ'
        'zigzag+turn YX'
        'zigzag+turn YZ'
        'zigzag+turn ZX'
        'zigzag+turn ZY'
    
    The '+turn' indicates that the whole pattern is repeated backwards with a
    rotation of steps[3] degrees, i.e.:
        1. Do pattern
        2. Rotate steps[3] degrees
        3. Do pattern backwards
    
    In case of a zigzag pattern, the first axis is the first to move (the
    long one).
    
    Example of 'zigzag XY':
        
                    y
                    ^
                    |
                    |
          y=ends[1] _       ___________stop
                    |       |
                    |       |___________
                    |                  |
                    |       ___________|
                    |       |
         y=steps[1] _       |___________
                    |                  |
                y=0 _  start___________|
                    |
                    |-------|----------|-----------> x
                           x=0        x=ends[0]
    
    Parameters
    ----------
    pattern : str
        Scanning pattern to generate. Available patterns are:
            'line on X'
            'line on Y'
            'line on Z'
            'line+turn on X'
            'line+turn on Y'
            'line+turn on Z'
            'zigzag XY'
            'zigzag XZ'
            'zigzag YX'
            'zigzag YZ'
            'zigzag ZX'
            'zigzag ZY'
            'zigzag+turn XY'
            'zigzag+turn XZ'
            'zigzag+turn YX'
            'zigzag+turn YZ'
            'zigzag+turn ZX'
            'zigzag+turn ZY'
        In case of a zigzag pattern, the first axis is the first to move (the
        long one).
    steps : list of floats
        Steps of every axis in the following order:
            steps[0] -> X
            steps[1] -> Y
            steps[2] -> Z
            steps[3] -> R
    ends : list of floats
        Maximum axis value of every axis in the following order:
            ends[0] -> X
            ends[1] -> Y
            ends[2] -> Z
            ends[3] -> R.

    Raises
    ------
    NotImplementedError
        For unimplemented or wrong patterns.

    Returns
    -------
    scanpatter : list
        List of strings describing the scanning pattern.

    Arnau, 01/02/2023
    '''
    if len(steps) != 4 or len(ends) != 4:
        print(f'steps and ends must have a length of 4, but lengths {len(steps)} and {len(ends)} where found.')
        return -1
    X_step, Y_step, Z_step, R_step = steps
    X_end, Y_end, Z_end, R_end = ends
    
    available_line_patterns = ['line on x', 'line on y', 'line on z', 'line+turn on x', 'line+turn on y', 'line+turn on z']
    available_zigzag_patterns = ['zigzag xy', 'zigzag xz', 'zigzag yx', 'zigzag yz', 'zigzag zx', 'zigzag zy',
                                 'zigzag+turn xy', 'zigzag+turn xz', 'zigzag+turn yx', 'zigzag+turn yz', 'zigzag+turn zx', 'zigzag+turn zy']
    
    if pattern.lower() in available_line_patterns:
        ax = pattern[-1].upper()
        if ax == 'X':
            end = X_end
            step = X_step
        elif ax == 'Y':
            end = Y_end
            step = Y_step
        elif ax == 'Z':
            end = Z_end
            step = Z_step
        
        if pattern.lower() in available_line_patterns[:3]:
            return _makeLinePattern(ax, step, end)
        else:
            scanpatter = _makeLinePattern(ax, step, end)
            reverse_scanpatter = _makeLinePattern(ax, -step, end)[::-1]
            return scanpatter + [f'R{R_step}'] + reverse_scanpatter
    
    elif pattern.lower() in available_zigzag_patterns:
        longaxis, shortaxis = pattern[-2].upper(), pattern[-1].upper()
        if longaxis == 'X':
            longend = X_end
            longstep = X_step
        elif longaxis == 'Y':
            longend = Y_end
            longstep = Y_step
        elif longaxis == 'Z':
            longend = Z_end
            longstep = Z_step
        
        if shortaxis == 'X':
            shortend = X_end
            shortstep = X_step
        elif shortaxis == 'Y':
            shortend = Y_end
            shortstep = Y_step
        elif shortaxis == 'Z':
            shortend = Z_end
            shortstep = Z_step
        
        if pattern.lower() in available_zigzag_patterns[:6]:
            return _makeZigZagPattern(f'{longaxis}{shortaxis}', longstep, shortstep, longend, shortend)
        else:
            scanpatter = _makeZigZagPattern(f'{longaxis}{shortaxis}', longstep, shortstep, longend, shortend)
            reverse_scanpatter = _makeZigZagPattern(f'{longaxis}{shortaxis}', -longstep, -shortstep, longend, shortend)[::-1]
            return scanpatter + [f'R{R_step}'] + reverse_scanpatter
    else:
        raise NotImplementedError

def _makeZigZagPattern(plane: str, longstep: float, shortstep: float, longend: float, shortend: float) -> list:
    '''
    Makes the zigzag pattern. The number of steps is always rounded down. See
    makeScanPattern(pattern, steps, ends) for more information.

    Parameters
    ----------
    plane : str
        Plane to perform zigzag on. Available planes are:
            'XY'
            'XZ'
            'YX'
            'YZ'
            'ZX'
            'ZY'
    longstep : float
        Step of first axis.
    shortstep : float
        Step of second axis.
    longend : float
        End value of first axis.
    shortend : float
        End value of second axis.

    Returns
    -------
    scanpatter : list
        List of strings describing the scanning pattern.

    Arnau, 01/02/2023
    '''
    longaxis, shortaxis = plane[0].upper(), plane[1].upper()
    Nlong = int(longend // abs(longstep))
    Nshort = int(shortend // abs(shortstep))
    scanpatter = [f'{longaxis}{longstep}'] * Nlong
    for i in range(1, Nshort+1):
        scanpatter += [f'{shortaxis}{shortstep}']
        scanpatter += [f'{longaxis}{(-1)**i * longstep}'] * Nlong
    return scanpatter

def _makeLinePattern(axis: str, step: float, end: float) -> list:
    '''
    Makes a linear pattern. The number of steps is always rounded down. See
    makeScanPattern(pattern, steps, ends) for more information.

    Parameters
    ----------
    axis : str
        Axis to perform the linear pattern on. Available axis are:
            'X'
            'Y'
            'Z'
            'R'
    step : float
        Step.
    end : float
        End value.

    Returns
    -------
    scanpatter : list
        List of strings describing the scanning pattern.

    Arnau, 01/02/2023
    '''
    return [f'{axis.upper()}{step}'] * int(end // abs(step))
