/*
Arduino UNO PinOut for SPI
--------------------------
D13 - CLK/SCLK/SCK (Clock)
D12 - MISO/SDI (Master In Slave Out / Slave Data In)
D11 - MOSI/SDO (Master Out Slave In / Slave Data Out)
D10 - CS/SS (Chip Select / Slave Select) --- Only in case Arduino board is the slave. In this case we are the Master, so we use D06.
D07 - DRDY/DRDYB (Data ready)

Callendar-Van Dusen equation is used: 
  R(T) = R0(1 + aT + bT^2 + c(T - 100)T^3)
Note:
  For T > 0ºC --> c = 0
*/

// INCLUDES
#include <SPI.h>

// CONSTANTS - Pins
static const int CSB1_pin = 6; // Chip Select (slave select) sensor 1, active low
static const int CSB2_pin = 5; // Chip Select (slave select) sensor 2, active low
static const int DRDYB_pin = 7; // Data Ready output, active low

// CONSTANTS - Registers
static const byte w_config_register = 0x80; // write config
static const byte r_config_register = 0x00; // read config
static const byte r_RTD_MSBs_register = 0x01; // read data (MSBs)
static const byte r_RTD_LSBs_register = 0x02; // read data (LSBs)

// CONSTANTS - Configuration
static const int Clock_Hz = 1000000; // 1 MHz
static const byte config = 0b11000001; // {1,1,0,0,0,0,0,1} = {V_BIAS=ON, ConversionMode=Auto, 1-SHOT=False, 3-Wire=False, FaultDetection=False, , FaultStatus=False, NotchFilter=50Hz}

// CONSTANTS - Temperature measurement
static const double R_REF1 = 399.5; // Reference Resistor installed on the board (sensor 1) - Ohm --- CALIBRATED from 400 Ohm
static const double R_REF2 = 399.4; // Reference Resistor installed on the board (sensor 2) - Ohm --- CALIBRATED from 400 Ohm
static const double RTD_at_0deg = 100; // RTD Resistance at 0 Degrees (Pt100) - Ohm
static const double a = 0.00390830; // Callendar-Van Dusen coefficient (1st order)
static const double b = -0.0000005775; // Callendar-Van Dusen coefficient (2nd order)
static const double c = -0.00000000000418301; // Callendar-Van Dusen coefficient (3rd and 4th order)
static const int ms = 100; // milliseconds between each temperature acquisition
static const int num_decimal_points = 4; // number of decimal point digits to pass over Serial communication

// VARIABLES
byte lsb_rtd;
byte msb_rtd;
double RTD;
double temperature1;
double temperature2;
byte fault_test;

// FUNCTIONS - Write data to register
void SPIwrite1(byte w_addr, byte data){
  SPI.beginTransaction(SPISettings(Clock_Hz, MSBFIRST, SPI_MODE3));
  digitalWrite(CSB1_pin, LOW);   
  SPI.transfer(w_addr);                 
  SPI.transfer(data);                
  digitalWrite(CSB1_pin, HIGH);
  SPI.endTransaction();
}
void SPIwrite2(byte w_addr, byte data){
  SPI.beginTransaction(SPISettings(Clock_Hz, MSBFIRST, SPI_MODE3));
  digitalWrite(CSB2_pin, LOW);   
  SPI.transfer(w_addr);                 
  SPI.transfer(data);                
  digitalWrite(CSB2_pin, HIGH);
  SPI.endTransaction();
}


// FUNCTIONS - Read data from register
byte SPIread1(byte r_addr){
  SPI.beginTransaction(SPISettings(Clock_Hz, MSBFIRST, SPI_MODE3));
  digitalWrite(CSB1_pin, LOW);   
  SPI.transfer(r_addr);                 
  byte value = SPI.transfer(0xFF);                
  digitalWrite(CSB1_pin, HIGH);
  SPI.endTransaction();
  return value;
}
byte SPIread2(byte r_addr){
  SPI.beginTransaction(SPISettings(Clock_Hz, MSBFIRST, SPI_MODE3));
  digitalWrite(CSB2_pin, LOW);   
  SPI.transfer(r_addr);                 
  byte value = SPI.transfer(0xFF);                
  digitalWrite(CSB2_pin, HIGH);
  SPI.endTransaction();
  return value;
}


// FUNCTIONS - Get temperature data
double getTemp1(){
  lsb_rtd = SPIread1(r_RTD_LSBs_register);  
  fault_test = lsb_rtd & 0x01;
  do{
    if(digitalRead(DRDYB_pin) == 0){
      msb_rtd = SPIread1(r_RTD_MSBs_register);
      RTD = ((msb_rtd << 7) + ((lsb_rtd & 0xFE) >> 1)); // Combining RTD_MSB and RTD_LSB to protray decimal value. Removing MSB and LSB during shifting/Anding
      RTD = (RTD * R_REF1) / 32768; // Conversion of ADC RTD code to resistance (15 bits = 2^15)
      temperature1 = -RTD_at_0deg*a + sqrt(RTD_at_0deg*RTD_at_0deg*a*a - 4*RTD_at_0deg*b*(RTD_at_0deg-RTD)); // Conversion of RTD resistance to Temperature
      temperature1 = temperature1/(2*RTD_at_0deg*b);
      lsb_rtd = SPIread1(r_RTD_LSBs_register);
      fault_test = lsb_rtd & 0x01;
    }
  } while(fault_test != 0);
  // Serial.println("Error was detected. The RTD resistance measured is not within the range specified in the Threshold Registers.");  
  return temperature1;
}
double getTemp2(){
  lsb_rtd = SPIread2(r_RTD_LSBs_register);  
  fault_test = lsb_rtd & 0x01;
  do{
    if(digitalRead(DRDYB_pin) == 0){
      msb_rtd = SPIread2(r_RTD_MSBs_register);
      RTD = ((msb_rtd << 7) + ((lsb_rtd & 0xFE) >> 1)); // Combining RTD_MSB and RTD_LSB to protray decimal value. Removing MSB and LSB during shifting/Anding
      RTD = (RTD * R_REF2) / 32768; // Conversion of ADC RTD code to resistance (15 bits = 2^15)
      temperature2 = -RTD_at_0deg*a + sqrt(RTD_at_0deg*RTD_at_0deg*a*a - 4*RTD_at_0deg*b*(RTD_at_0deg-RTD)); // Conversion of RTD resistance to Temperature
      temperature2 = temperature2/(2*RTD_at_0deg*b);
      lsb_rtd = SPIread2(r_RTD_LSBs_register);
      fault_test = lsb_rtd & 0x01;
    }
  } while(fault_test != 0);
  // Serial.println("Error was detected. The RTD resistance measured is not within the range specified in the Threshold Registers.");
  return temperature2;
}


// --------------
// SETUP and LOOP
// --------------
void setup() {
  Serial.begin(9600);
  pinMode(CSB1_pin, OUTPUT);
  pinMode(CSB2_pin, OUTPUT);
  pinMode(DRDYB_pin, INPUT);
  SPI.begin();

  SPIwrite1(w_config_register, config); // we do it several times just in case
  SPIwrite1(w_config_register, config);
  SPIwrite1(w_config_register, config);

  SPIwrite2(w_config_register, config); // we do it several times just in case
  SPIwrite2(w_config_register, config);
  SPIwrite2(w_config_register, config);
}

void loop() {
  temperature1 = getTemp1(); // Calling getTemp1() to read RTD registers and convert to Temperature reading
  temperature2 = getTemp2(); // Calling getTemp2() to read RTD registers and convert to Temperature reading

  Serial.print(temperature1, num_decimal_points);
  Serial.print(" ");
  Serial.print(temperature2, num_decimal_points);
  Serial.print("\n");
  delay(ms); // Sets the speed of acquisition
}
