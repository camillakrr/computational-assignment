import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QGridLayout, QSizePolicy
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import random
import numpy as np

limit = 1.1

def f(x, y):
    try:
        return (1 - 2*y)*np.exp(x) + y*y + np.exp(2*x)
    except FloatingPointError:
        print("Double overflow raised. Plot cannot be displayed")
        sys.exit(0)

def exactConstant(x, y):
    return (1 - x*(np.exp(x) - y))/(np.exp(x) - y)

def getProperX(xArr, const, n, h):
    l, r = -1, n-1
    while (r - l > 1):
        m = (l + r) // 2
        if xArr[m] > -const + h/limit:
            r = m
        else:
            l = m
        #print(l)
        #print(r)

    return r


def exactSolution(x, y, X, step):
    xArr = np.linspace(x, X, step)
    yArr = [np.inf for i in range(step)]
    yArr[0] = y
    constant = exactConstant(x, y)
    i = 1

    while i < step:
        if xArr[i] > -constant - (X-x)/(step-1)/limit and xArr[i] < -constant + (X-x)/(step-1)/limit:
            escape = getProperX(xArr, constant, len(xArr), (X-x)/(step-1))
            i = escape + 1

        yArr[i] = np.exp(xArr[i]) - 1/(xArr[i] + constant)
        #print(yArr[i])
        i += 1
    
    return [xArr, yArr]

def improvedEulerSolution(x, y, X, step):
    dx = (X-x)/(step-1)
    xArr = np.linspace(x, X, step)
    yArr = [np.inf for i in range(step)]
    yArr[0] = y
    exact = exactSolution(x, y, X, step)
    constant = exactConstant(x, y)
    yExact = exact[1]
    i = 1

    while i < step:
        if x + dx*i > -constant - dx/limit and x + dx*i < -constant + dx/limit:
            escape = getProperX(xArr, constant, len(xArr), dx)
            yArr[escape] = yExact[escape]
            if yExact[escape] == np.inf:
                yArr[escape] = yExact[escape+1]
            i = escape + 1
            continue

        yArr[i] = dx*f(xArr[i-1] + dx/2, yArr[i-1] + dx/2*f(xArr[i-1], yArr[i-1])) + yArr[i-1]
        i += 1

    return [xArr, yArr]

def eulerSolution(x, y, X, step):
    dx = (X-x)/(step-1)
    xArr = np.linspace(x, X, step)
    yArr = [np.inf for i in range(step)]
    yArr[0] = y
    exact = exactSolution(x, y, X, step)
    constant = exactConstant(x, y)
    yExact = exact[1]
    i = 1

    while i < step:
        if x + dx*i > -constant - dx/limit and x + dx*i < -constant + dx/limit:
            escape = getProperX(xArr, constant, len(xArr), dx)
            yArr[escape] = yExact[escape]
            if yExact[escape] == np.inf:
                yArr[escape] = yExact[escape+1]
            i = escape + 1
            #print("Entered zone %f" % yArr[escape])
            #print(escape)
            #print('-----------------------')
            continue

        #print(i)
        #print(f(xArr[i-1], yArr[i-1]))
        yArr[i] = dx*(f(xArr[i-1], yArr[i-1])) + yArr[i-1]
        #print(xArr[i])
        #print(yArr[i])
        yArr[i] = round(yArr[i], 3)
        i = i + 1
    
    return [xArr, yArr]

def rungeKuttaCoefficients(h, x, y):
    y1 = h*f(x, y)
    y2 = h*f(x + h/2, y + y1/2)
    y3 = h*f(x + h/2, y + y2/2)
    y4 = h*f(x + h, y + y3)
    return [y1, y2, y3, y4]

def rungeKuttaSolution(x, y, X, step):
    dx = (X-x)/(step-1)
    xArr = np.linspace(x, X, step)
    yArr = [np.inf for i in range(step)]
    yArr[0] = y
    exact = exactSolution(x, y, X, step)
    constant = exactConstant(x, y)
    yExact = exact[1]

    coefficients = rungeKuttaCoefficients(dx, x, y)
    y1, y2, y3, y4 = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
    #print(coefficients)
    i = 1

    while i < step:
        if x + dx*i > -constant - dx/limit and x + dx*(i-1) < -constant + dx/limit:
            escape = getProperX(xArr, constant, len(xArr), dx)
            if yExact[escape] == np.inf:
                yArr[escape] = yExact[escape+1]
            coefficients = rungeKuttaCoefficients(dx, xArr[escape], yArr[escape])
            y1, y2, y3, y4 = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
            i = escape + 1
            continue

        yArr[i] = yArr[i-1] + (y1 + 2*y2 + 2*y3 + y4)/6
        coefficients = rungeKuttaCoefficients(dx, xArr[i], yArr[i])
        #print(coefficients)
        y1, y2, y3, y4 = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
        i = i + 1

    return [xArr, yArr]

def localError(x, y, X, step, numerical):
    exact = exactSolution(x, y, X, step)
    error = []
    
    for i in range(len(exact[0])):
        error.append(abs(numerical[1][i]-exact[1][i]))
    
    return error

def globalError(x, y, X, stepL, stepR, numerical):
    xArr = [i for i in range(stepL, stepR+1)]
    yArr = []

    for i in range(stepL, stepR+1):
        local = localError(x, y, X, i, numerical(x, y, X, i))
        mx = -np.inf
        for err in local:
            if err != np.inf:
                mx = max(mx, err)
        yArr.append(mx)
    
    return [xArr, yArr]

class DiffEq(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.initUIMain()
        self.btn.clicked.connect(self.plotButton)
        self.btn_error.clicked.connect(self.errorButton)
        self.btn_global.clicked.connect(self.globalButton)
        self.setGeometry(300, 300, 1366, 1024)
        self.setFixedSize(self.size())
        self.setWindowTitle('Differential Equations Plotting')
        self.show()

    def initUIMain(self):
        self.btn = QPushButton('Plot', self)
        self.btn.setToolTip("Let's plot!")
        self.btn.resize(self.btn.sizeHint())

        self.btn_error = QPushButton('Errors', self)
        self.btn_error.setToolTip("Errors plotting")
        self.btn_error.resize(self.btn_error.sizeHint())

        self.btn_global = QPushButton('Globals', self)
        self.btn_global.setToolTip('Global errors')
        self.btn_global.resize(self.btn_global.sizeHint())

        xleftLabel = QLabel('x0')
        xrightLabel = QLabel('X')
        yinitLabel = QLabel('y0')
        stepLabel = QLabel('step')
        stepRangeLabel = QLabel('step range')

        self.xleftEdit = QLineEdit()
        self.xrightEdit = QLineEdit()
        self.yinitEdit = QLineEdit()
        self.stepEdit = QLineEdit()
        self.stepL = QLineEdit()
        self.stepR = QLineEdit()
        self.plotSolution = PlotCanvas(self)
        self.plotToolbar = NavigationToolbar(self.plotSolution, self)

        self.xleftEdit.setFixedWidth(120)
        self.xrightEdit.setFixedWidth(120)
        self.yinitEdit.setFixedWidth(120)
        self.stepEdit.setFixedWidth(120)
        self.stepL.setFixedWidth(120)
        self.stepR.setFixedWidth(120)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(xleftLabel, 1, 0)
        grid.addWidget(self.xleftEdit, 1, 1)
        grid.addWidget(xrightLabel, 2, 0)
        grid.addWidget(self.xrightEdit, 2, 1)
        grid.addWidget(yinitLabel, 3, 0)
        grid.addWidget(self.yinitEdit, 3, 1)
        grid.addWidget(stepLabel, 4, 0)
        grid.addWidget(self.stepEdit, 4, 1)
        grid.addWidget(stepRangeLabel, 6, 1)
        grid.addWidget(self.stepL, 6, 2)
        grid.addWidget(self.stepR, 7, 2)
        grid.addWidget(self.plotSolution, 1, 2, 4, 1)
        grid.addWidget(self.btn, 5, 0)
        grid.addWidget(self.btn_error, 5, 1)
        grid.addWidget(self.plotToolbar, 5, 2)
        grid.addWidget(self.btn_global, 6, 0)

        self.setLayout(grid)

    def plotButton(self):
        x0 = int("%s" % self.xleftEdit.text())
        X = int("%s" % self.xrightEdit.text())
        y0 = int("%s" % self.yinitEdit.text())
        step = int("%s" % self.stepEdit.text())
        #print([x0, X, y0, step])
        
        exact = exactSolution(x0, y0, X, step)
        euler = eulerSolution(x0, y0, X, step)
        improvedEuler = improvedEulerSolution(x0, y0, X, step)
        rungeKutta = rungeKuttaSolution(x0, y0, X, step)

        xExact, yExact = exact[0], exact[1]
        xEuler, yEuler = euler[0], euler[1]
        xImprovedEuler, yImprovedEuler = improvedEuler[0], improvedEuler[1]
        xRungeKutta, yRungeKutta = rungeKutta[0], rungeKutta[1]

        #print(yEuler)

        print(yExact)
        #print(yEuler)
        #print(yImprovedEuler)

        self.plotSolution.ax.clear()
        self.plotSolution.plot(xExact, yExact, 'r-', 'Exact solution')
        self.plotSolution.plot(xEuler, yEuler, 'b-', 'Euler')
        self.plotSolution.plot(xImprovedEuler, yImprovedEuler, 'g-', 'Improved Euler')
        self.plotSolution.plot(xRungeKutta, yRungeKutta, 'k-', 'Runge-Kutta')

    def errorButton(self):
        x0 = int("%s" % self.xleftEdit.text())
        X = int("%s" % self.xrightEdit.text())
        y0 = int("%s" % self.yinitEdit.text())
        step = int("%s" % self.stepEdit.text())

        exact = exactSolution(x0, y0, X, step)
        eulerError = localError(x0, y0, X, step, eulerSolution(x0, y0, X, step))
        improvedEulerError = localError(x0, y0, X, step, improvedEulerSolution(x0, y0, X, step))
        rungeKuttaError = localError(x0, y0, X, step, rungeKuttaSolution(x0, y0, X, step))

        self.plotSolution.ax.clear()
        self.plotSolution.plot(exact[0], eulerError, 'b-', 'Euler local error')
        self.plotSolution.plot(exact[0], improvedEulerError, 'g-', 'Improved Euler local error')
        self.plotSolution.plot(exact[0], rungeKuttaError, 'k-', 'Runge-Kutta local error')

    def globalButton(self):
        x0 = int("%s" % self.xleftEdit.text())
        X = int("%s" % self.xrightEdit.text())
        y0 = int("%s" % self.yinitEdit.text())
        stepL = int("%s" % self.stepL.text())
        stepR = int("%s" % self.stepR.text())

        eulerGlobalErr = globalError(x0, y0, X, stepL, stepR, eulerSolution)
        improvedEulerGlobalErr = globalError(x0, y0, X, stepL, stepR, improvedEulerSolution)
        rungeKuttaGlobalErr = globalError(x0, y0, X, stepL, stepR, rungeKuttaSolution)

        self.plotSolution.ax.clear()
        print(eulerGlobalErr[0])
        print(eulerGlobalErr[1])
        self.plotSolution.plot(eulerGlobalErr[0], eulerGlobalErr[1], 'b-', 'Euler global error')
        self.plotSolution.plot(eulerGlobalErr[0], improvedEulerGlobalErr[1], 'g-', 'Improved Euler global error')
        self.plotSolution.plot(eulerGlobalErr[0], rungeKuttaGlobalErr[1], 'k-', 'Runge-Kutta global error')


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(r'$(1-2y)e^x + y^2 + e^{2x}$')
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()
 
 
    def plot(self, x=[], y=[], color='', _label=''):
        self.ax.plot(x, y, color, label=_label)
        self.ax.legend()
        self.draw()

if __name__ == '__main__':
    np.seterr(all='raise')
    app = QApplication(sys.argv)
    diffeq = DiffEq()
    sys.exit(app.exec_())