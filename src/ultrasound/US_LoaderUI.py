# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'US_LoaderUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#

from PyQt5 import QtCore, QtWidgets
import glob
from . import US_Loaders as USL

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(603, 189)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(60, 20, 451, 135))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.GridLayout_LoadData = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.GridLayout_LoadData.setContentsMargins(0, 0, 0, 0)
        self.GridLayout_LoadData.setObjectName("GridLayout_LoadData")
        
        # LineEdit_WP
        self.LineEdit_WP = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.LineEdit_WP.setObjectName("LineEdit_WP")
        self.GridLayout_LoadData.addWidget(self.LineEdit_WP, 1, 2, 1, 1)
        
        # Label_Excitation
        self.Label_Excitation = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Label_Excitation.setObjectName("Label_Excitation")
        self.GridLayout_LoadData.addWidget(self.Label_Excitation, 2, 0, 1, 1)
        
        # Label_Data
        self.Label_Data = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Label_Data.setObjectName("Label_Data")
        self.GridLayout_LoadData.addWidget(self.Label_Data, 0, 0, 1, 1)
        
        # Label_WP
        self.Label_WP = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Label_WP.setObjectName("Label_WP")
        self.GridLayout_LoadData.addWidget(self.Label_WP, 1, 0, 1, 1)
        
        # Button_Load
        self.Button_Load = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Button_Load.setObjectName("Button_Load")
        self.GridLayout_LoadData.addWidget(self.Button_Load, 3, 0, 1, 3)
        self.Button_Load.clicked.connect(lambda: self.Load())
        
        # ComboBox_Excitation
        self.ComboBox_Excitation = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.ComboBox_Excitation.setObjectName("ComboBox_Excitation")
        self.GridLayout_LoadData.addWidget(self.ComboBox_Excitation, 2, 1, 1, 1)
        
        # Button_Browse_WP
        self.Button_Browse_WP = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Button_Browse_WP.setObjectName("Button_Browse_WP")
        self.GridLayout_LoadData.addWidget(self.Button_Browse_WP, 1, 1, 1, 1)
        self.Button_Browse_WP.clicked.connect(lambda: self.Browse_WP())
        
        # Button_Browse_Data
        self.Button_Browse_Data = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Button_Browse_Data.setObjectName("Button_Browse_Data")
        self.GridLayout_LoadData.addWidget(self.Button_Browse_Data, 0, 1, 1, 1)
        self.Button_Browse_Data.clicked.connect(lambda: self.Browse_Data())
        
        # LineEdit_Data
        self.LineEdit_Data = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.LineEdit_Data.setObjectName("LineEdit_Data")
        self.GridLayout_LoadData.addWidget(self.LineEdit_Data, 0, 2, 1, 1)
        
        # Label_LoadSuccessful
        self.Label_LoadSuccessful = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Label_LoadSuccessful.setObjectName("Label_LoadSuccessful")
        self.GridLayout_LoadData.addWidget(self.Label_LoadSuccessful, 4, 0, 1, 3)
        self.Label_LoadSuccessful.setAlignment(QtCore.Qt.AlignCenter)
        
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.PE_Ascan = None
        self.TT_Ascan = None
        self.WP_Ascan = None
        self.stdVar = None
        self.ScanLen = None

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Load Data"))
        self.Label_Excitation.setText(_translate("MainWindow", "Excitation"))
        self.Label_Data.setText(_translate("MainWindow", "Data"))
        self.Label_WP.setText(_translate("MainWindow", "Water Path"))
        self.Label_LoadSuccessful.setText(_translate("MainWindow", ""))
        self.Button_Load.setText(_translate("MainWindow", "Load"))
        self.Button_Browse_WP.setText(_translate("MainWindow", "Browse"))
        self.Button_Browse_Data.setText(_translate("MainWindow", "Browse"))


    def Browse_Data(self):
        folder_path = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.LineEdit_Data.setText(folder_path)
        
        gencode_paths = glob.glob(folder_path+'/gencode*.txt')
        for i in range(len(gencode_paths)):
            self.ComboBox_Excitation.addItem(str(i+1))

    def Browse_WP(self):
        folder_path = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.LineEdit_WP.setText(folder_path)

    def Load(self):
        WP_Path = self.LineEdit_WP.text()
        Specimen_Path = self.LineEdit_Data.text()
        GenCode = self.ComboBox_Excitation.currentText()
        
        # load experiment variables from stdVar
        self.stdVar = USL.StdVar(Specimen_Path + r'\standard.var')

        # beware!! in this experiment, Ascans where accumulated in acq. stage, but
        # not divided by the number of averages, that must be extracted from stdVar
        Avg = 1 # number of Ascans to be read and averaged

        self.ScanLen = int(self.stdVar.Smax-self.stdVar.Smin) # length of Ascans

        filename_Ch2 = Specimen_Path + r'\ScanADCgenCod' + str(GenCode) + 'Ch2.bin' # load PE
        self.PE_Ascan = USL.LoadBinAscan(filename_Ch2, Avg, self.ScanLen, N1=0, N2=0) / self.stdVar.AvgSamplesNumber

        filename_Ch1 = Specimen_Path + r'\ScanADCgenCod' + str(GenCode) + 'Ch1.bin' # load TT 
        self.TT_Ascan = USL.LoadBinAscan(filename_Ch1, Avg, self.ScanLen, N1=0, N2=0) / self.stdVar.AvgSamplesNumber

        filename_WP = WP_Path + r'\ScanADCgenCod' + str(GenCode) + 'Ch1.bin' # load WP
        self.WP_Ascan = USL.LoadBinAscan(filename_WP, Avg, self.ScanLen, N1=0, N2=0) / self.stdVar.AvgSamplesNumber
        
        self.Label_LoadSuccessful.setText("Load Successful!")
        QtCore.QTimer.singleShot(2000, self.hideLabel)

    def hideLabel(self):
        self.Label_LoadSuccessful.setText("")

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def LoaderUI_getVars(ui):
    return ui.PE_Ascan,  ui.TT_Ascan, ui.WP_Ascan, ui.ScanLen, ui.stdVar




def LoaderUI():
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    return MainWindow, ui

if __name__ == "__main__":
    if run_from_ipython():
        MainWindow, ui = LoaderUI()
    else:
        import sys
        app = QtWidgets.QApplication(sys.argv)
        MainWindow, ui = LoaderUI()
        sys.exit(app.exec_())
        

