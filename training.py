from itertools import groupby
from datetime import datetime as dt
from analyzer import plot,groupByWeek, groupByHour,groupByDay,toNumHours

STARTTIME = dt(2012,1,1,0,0,0)

def aug(x):
    return map(lambda e: (1,) + e, x )

def trimLst(groupByWeek):
    unabrdg = zip(*groupByWeek)
    del(unabrdg[-1])
    return zip(*unabrdg)

def dummyWkd(timestamps):
    wkds = range(1,8)
    wkdFtr = map(lambda x : x.isocalendar()[2], timestamps)
    return map(lambda x: map(lambda y: 1 if x == y else 0, wkds), wkdFtr)

def dummyHr(timestamps):
    hrs = range(1,25)
    hrFtr = map(lambda x :x.hour, timestamps)
    return map(lambda x: map(lambda y: 1 if x == y else 0, hrs), hrFtr)

# features is a list of data points, each data point is a tuple of binary features
# label are the labels of the data points
# data and label should be of same length
def fturizWkdHr(timestamps):
    return map(lambda x,y : tuple(x + y), dummyWkd(timestamps), dummyHr(timestamps))

def fturizWk(timestamps):
    return map(lambda x : (x,), toNumHours(timestamps))

class Trainer(object):
    # data is list of timestamps in json format
    def __init__(self, app):
        self.trained = False
        self.app = app
        self.logger = app.logger

    def train(self, timestamps):
        return self.trainLin(timestamps)

    def predict(self, timestamps):
        return map(lambda x:x if x > 0 else 0, self.predictLin(timestamps))

    def trainLin(self, timestamps):
        metrics = ''
        weeklyData = groupByWeek(timestamps)
        trimWkData = trimLst(weeklyData)
        wkFtr = fturizWk(trimWkData[0])
        self.wkMdl = self.linReg(wkFtr, trimWkData[1])
        if self.logger:
            self.logger.info(self.wkMdl.coef_)
        metrics += 'R-sqr = {:.3f}, coef = {}\n'.format(self.wkMdl.score(aug(wkFtr), trimWkData[1]), self.wkMdl.coef_)

        hourlydata = groupByHour(timestamps)
        adjHrCnt = self.adjWkTrend(hourlydata)
        wkdHrFtr = fturizWkdHr(hourlydata[0])
        self.wkdHrMdl = self.lasso(wkdHrFtr, adjHrCnt)
        if self.logger:
            self.logger.info(self.wkdHrMdl.coef_)
        metrics += 'R-sqr = {:.3f}, coef = {}\n'.format(self.wkdHrMdl.score(aug(wkdHrFtr), adjHrCnt), self.wkdHrMdl.coef_)

        self.trained = True
        img1,img2 = self.makePlot(weeklyData,hourlydata)
        return img1,img2,metrics

    def adjWkTrend(self, hourlydata):
        hourlyPredict = self.calcWkFctr(hourlydata[0])
        return map(lambda x,y : x/y, hourlydata[1], hourlyPredict)

    def calcWkFctr(self, timestamps):
        inputdata = fturizWk(timestamps)
        return self.wkMdl.predict(aug(inputdata))

    def calcDayHrFctr(self, timestamps):
        inputdata = fturizWkdHr(timestamps)
        return self.wkdHrMdl.predict(aug(inputdata))

    def predictLin(self, timestamps):
        if not (self.trained):
            return None
        return map(lambda x,y : x * y, self.calcDayHrFctr(timestamps), self.calcWkFctr(timestamps))

    @staticmethod
    def linReg(x, y):
        from sklearn import linear_model
        clf = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
        clf.fit (aug(x), y)
        return clf

    @staticmethod
    def lasso(x, y):
        from sklearn import linear_model
        clf = linear_model.Ridge(alpha=0.01)
        clf.fit (aug(x), y)
        return clf

    def makePlot(self, weeklyData, hourlydata):
        import time
        imgname = time.strftime("%Y%m%d-%H%M%S")
        img1,img2 = imgname+'1.png',imgname+'2.png'
        plt = plot([
            {'data' : weeklyData, 'title' : 'Weekly training error', 'ylabel' : 'Request count', 'label' : 'Original data'},
            {'data' : [weeklyData[0],self.calcWkFctr(weeklyData[0])], 'xlabel' : 'Hour', 'label' : 'Trained result'},
            ])
        plt.savefig('resources/'+img1, bbox_inches='tight')

        plt = plot([
            {'data' : hourlydata, 'title' : 'Overall demand curve training error', 'ylabel' : 'Request count', 'label' : 'Original data'},
            {'data' : [hourlydata[0], self.predict(hourlydata[0])], 'xlabel' : 'Hour', 'label' : 'Trained result'},
            ])
        plt.savefig('resources/'+img2, bbox_inches='tight')
        return img1,img2
        
if __name__ == '__main__':
    clf = Trainer.linReg([9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [1602, 2200, 2074, 2351, 2480, 2258, 2770, 3314, 3154, 244])
    print clf.predict(aug([9,10]))
