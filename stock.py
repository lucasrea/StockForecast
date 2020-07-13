import yfinance as yf
import datetime
import os
import pandas as pd
import numpy as np
from finta import TA
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



spark = SparkSession.builder.appName('stockanalysis').getOrCreate()


class Ticker():

    NUM_DAYS = 1000     # The number of days of historical data to retrieve
    INTERVAL = '1d'     # Sample rate of historical data

    # List of symbols for technical indicators
    INDICATORS = ['RSI', 'MACD', 'STOCH', 'ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']


    def __init__(self, symbol):

        """
        Constructor for class
        Will obtain historical data for NUM_DAYS number of days
        :param symbol: ticker of stock
        """

        self.symbol = symbol
        self._get_historical_data()

    def _get_historical_data(self):

        """
        Function that uses the yfinance API to get stock data
        :return:
        """

        start = (datetime.date.today() - datetime.timedelta( self.NUM_DAYS) )
        end = datetime.datetime.today()

        self.data = yf.download(self.symbol, start=start, end=end, interval=self.INTERVAL)
        self.data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)

    def _exponential_smooth(self, alpha):

        """
        Function that exponentially smooths dataset so values are less 'rigid'
        :param alpha: weight factor to weight recent values more
        """

        self.data = self.data.ewm(alpha=alpha).mean()

    def _get_indicator_data(self):

        """
        Function that uses the finta API to calculate technical indicators used as the features
        :return:
        """

        for indicator in self.INDICATORS:
            ind_data = eval('TA.' + indicator + '(self.data)')
            if not isinstance(ind_data, pd.DataFrame):
                ind_data = ind_data.to_frame()
            self.data = self.data.merge(ind_data, left_index=True, right_index=True)
        self.data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

        # Also calculate moving averages for features
        self.data['ema50'] = self.data['close'] / self.data['close'].ewm(50).mean()
        self.data['ema21'] = self.data['close'] / self.data['close'].ewm(21).mean()
        self.data['ema14'] = self.data['close'] / self.data['close'].ewm(14).mean()
        self.data['ema5'] = self.data['close'] / self.data['close'].ewm(5).mean()

        # Remove columns that won't be used as features
        del (self.data['open'])
        del (self.data['high'])
        del (self.data['low'])
        del (self.data['volume'])
        del (self.data['Adj Close'])

    def _produce_prediction(self, window=10):

        """
        Function that produces the 'truth' values
        At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
        :param window: number of days, or rows to look ahead to see what the price did
        """

        prediction = (self.data.shift(-window)['close'] >= self.data['close'])
        prediction = prediction.iloc[:-window]
        self.data['pred'] = prediction.astype(int)

    def _produce_data(self, window):

        """
        Main data function that calls the others to smooth, get features, and create the predictions
        :param window: value used to determine the prediction
        :return:
        """

        self._exponential_smooth(0.9)
        self._get_indicator_data()
        self._produce_prediction(window=window)

        del (self.data['close'])
        self.data = self.data.dropna()

    def _split_data(self):

        """
        Function to partition the data into the train and test set
        :return:
        """

        self.y = self.data['pred']
        features = [x for x in self.data.columns if x not in ['pred']]
        self.X = self.data[features]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size= 2 * len(self.X) // 3)

    def _train_random_forest(self):

        """
        Function that uses random forest classifier to train the model
        :return:
        """

        rf = RandomForestClassifier(n_jobs=-1, n_estimators=85, random_state=65)
        rf.fit(self.X_train, self.y_train.values.ravel())
        prediction = rf.predict(self.X_test)

        print(classification_report(self.y_test, prediction))
        print(confusion_matrix(self.y_test, prediction))
        print(rf.feature_importances_)


    def _data_clean(self, x=15):

        t1 = time.time()
        self._produce_data(window=x)
        self._split_data()
        print(str(time.time() - t1) + ' seconds to clean data')


    def _model(self):

        t1 = time.time()
        self._train_random_forest()
        print(time.time() - t1)

    def _spark_rf(self):
        self.df = spark.createDataFrame(self.data)

        features = []
        for col in self.df.columns:
            if col == 'pred':
                continue
            else:
                features.append(col)

        (trainingData, testData) = self.df.randomSplit([0.7, 0.3], seed=24234232)

        assembler = VectorAssembler(inputCols=features, outputCol="features")
        #rf = RandomForestClassifier(labelCol="pred", featuresCol="features", numTrees=500)
        gbt = gbt = GBTClassifier(labelCol="pred", featuresCol="features", maxIter=200)
        pipeline = Pipeline(stages=[assembler, gbt])

        model = pipeline.fit(trainingData)
        predictions = model.transform(testData)

        # Select (prediction, true label) and compute test error
        evaluator = MulticlassClassificationEvaluator(
            labelCol="pred", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Test Error = %g" % (1.0 - accuracy))


t = Ticker('SPY')
t._data_clean()
t._spark_rf()


#t._model()




