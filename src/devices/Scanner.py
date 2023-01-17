# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 08:46:32 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
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
        self.uStepR = 1.8 # (200steps=360deg) --- the other value that doesnt work: 0.0281

        # Current coordinates
        self.readCoords()
        
        # Enable all by default
        self.enableAll()

        # Set default directions to: '+', '+', '-', '+'
        self.setDirections()

        # Set default speedtypes to 'rectangular'
        self.setSpeedtypes()
    
        # Set default speeds (mm/s): 100, 100, 100, 10
        self.setSpeeds()
        

    def getuSteps(self, axis):
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

    def readCoords(self):
        return self.X, self.Y, self.Z, self.R


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
        

    # ======== SET SPEED ========
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


    # ======== SET RAMPING SPEED ========
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
    

    # ======== SET RANDOM SPEED PARAMETER ========
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


    # ======== OPEN, CLOSE, STOP, ENABLE ========
    def close(self):
        self.ser.close()

    def open(self):
        try:
            self.ser.open()
            print(f'Scanner {self.port}> Port opened')
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



def _parseSpeedtype(speedtype):
    allowed_speedtypes_str = ['rectangular', 'gaussian', 'random', 'triangle']
    if isinstance(speedtype, str):
        speedtype_lower = speedtype.lower()
        if speedtype_lower in allowed_speedtypes_str:
            code = allowed_speedtypes_str.index(speedtype_lower)
    elif speedtype in [0,1,2,3]:
        code = speedtype
    return code     
        
        
