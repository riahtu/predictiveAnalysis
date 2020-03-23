from PredictionAlgorithms.SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities
import pandas as pd
from pyspark.sql.functions import lit,when,col
from pyspark.sql import SparkSession

'''temporary'''
sparkTest = \
    SparkSession.builder.appName('DMXPredictiveAnalytics').master('local[*]').getOrCreate()
sparkTest.sparkContext.setLogLevel('ERROR')


class LexiconSentimentAnalysis(SentimentAnalysis):
    global spark

    def sentimentAnalysis(self, sentimentInfoData):
        '''requirement--
        1.- sentiment sentence containing colmName-
        2.- sentiment dataset parquet path
        3.- positive dictionary parquet path
        4.- negative dictionary parquet path
        5.- positive and negative dictionary colm name containing words/sentiment. '''
        sparkSession = sentimentInfoData.get(pc.SPARK)
        global spark
        spark = sparkSession

        datasetPath = sentimentInfoData.get(pc.SENTIMENTDATASETPATH)
        dataset = spark.read.parquet(datasetPath)
        dataset = pu.addInternalId(dataset)
        sentimentInfoData.update({pc.DATASET: dataset})
        sentimentDataset = self.textPreProcessing(sentimentInfoData)
        textProcessing = TextProcessing()
        posNegDataset = textProcessing.mergePosNegDataset(sentimentInfoData)
        dataset = self.addTag(sentimentDataset, pc.DMXSTEMMEDWORDS, posNegDataset)

    '''additional functionality will be written later after this.'''
    def createTaggedDataset(self, dataset, indexList, taggedRowList, positiveNum, negativeNum, totalNum, sentimentScores):
        zipData = zip(indexList, taggedRowList, positiveNum, negativeNum, totalNum, sentimentScores)
        columnList = [pc.DMXINDEX, pc.DMXTAGGEDCOLM, pc.POSITIVENUM,
                      pc.NEGATIVENUM, pc.TOTALWORDS, pc.SENTIMENTSCORE]
        pandasDataframe = pd.DataFrame(zipData, columns=columnList)
        taggedDataset = spark.createDataFrame(pandasDataframe)
        dataset = PredictiveUtilities.joinDataset(dataset, taggedDataset, pc.DMXINDEX)
        dataset = self.dropNeutral(dataset)
        dataset = self.performSentimentAnalysis(dataset)

        return dataset

    def dropNeutral(self, dataset):
        positiveColm = pc.POSITIVENUM
        negativeColm = pc.NEGATIVENUM
        query = "!(" + positiveColm + " == 0 AND " + negativeColm + " == 0)"
        dataset = dataset.filter(query)
        return dataset

    def addTag(self,dataset,colName,sentimentDictionary):
        taggedRowList = []
        indexList = []
        positiveNum = []
        negativeNum = []
        totalNum = []
        sentimentScores = []
        for index, row in enumerate(dataset.select(pc.DMXINDEX, colName).rdd.toLocalIterator()):
            dmxIndex = row[0]
            rowList = row[1]
            sentimentData = self.createSentimentData(rowList, sentimentDictionary)

            taggedRow = sentimentData.get(pc.SENTIMENTROW)
            posNum = sentimentData.get(pc.POSITIVENUM)
            negNum = sentimentData.get(pc.NEGATIVENUM)
            totalWords = sentimentData.get(pc.TOTALWORDS)
            sentimentScore = sentimentData.get(pc.SENTIMENTSCORE)

            indexList.append(dmxIndex)
            taggedRowList.append(taggedRow)
            positiveNum.append(posNum)
            negativeNum.append(negNum)
            totalNum.append(totalWords)
            sentimentScores.append(sentimentScore)
            print(sentimentScore)

        '''create dataset of the taggedColm -- & join the tagged dataset with the original dataset'''
        dataset = self.createTaggedDataset(dataset, indexList, taggedRowList, positiveNum, negativeNum, totalNum,
                                           sentimentScores)

        return dataset

    def createSentimentData(self, sentimentRow, posNegDataset):
        isRowUpdate = False
        posNum = 0
        negNum = 0
        totalWords = len(sentimentRow)
        for posNegIndex, row in enumerate(
                posNegDataset.select(pc.DMXINDEX, pc.DMXDICTIONARYCOLNAME, pc.DMXSENTIMENT).rdd.toLocalIterator()):
            compareText = row[pc.DMXDICTIONARYCOLNAME]  # need to get the colm Name from user
            sentiment = row[pc.DMXSENTIMENT]
            if (sentimentRow.__contains__(compareText)):
                isRowUpdate = True
                for index, text in enumerate(sentimentRow):
                    if (compareText.__eq__(text)):
                        if (sentiment.__eq__(pc.DMXPOSITIVE)):
                            posNum = posNum + 1
                        elif (sentiment.__eq__(pc.DMXNEGATIVE)):
                            negNum = negNum + 1
                        sentimentRow[index] = text + "[" + sentiment + "]"

        # calculate the no of positive words, negative words, and total number of words, and sentiment score.
        if (isRowUpdate):
            sentimentScore = round((posNum - negNum) / totalWords, 4)
        else:
            sentimentScore = 0.0

        sentimentData = {
            pc.SENTIMENTROW: sentimentRow,
            pc.POSITIVENUM: posNum,
            pc.NEGATIVENUM: negNum,
            pc.TOTALWORDS: totalWords,
            pc.SENTIMENTSCORE: sentimentScore
        }

        return sentimentData

    def performSentimentAnalysis(self, dataset):
        sentimentScoreMean = float(list(dataset.select(pc.SENTIMENTSCORE)
                                        .summary("mean").toPandas()[pc.SENTIMENTSCORE])[0])
        print("sentiment-mean:- ", sentimentScoreMean)
        dataset = dataset.withColumn(pc.SENTIMENTVALUE, when(col(pc.SENTIMENTSCORE) > sentimentScoreMean,
                                                               pc.POSITIVE).otherwise(pc.NEGATIVE))
        dataset.groupby(pc.SENTIMENTVALUE).count().show()  # just for testing purpose only.
        return dataset

if (__name__ == "__main__"):
    reviewDatasetPath = "/home/fidel/Documents/MOVIEDATASETPARQUET.parquet"
    positiveDatasetPath = "/home/fidel/Documents/POSITIVESENTIMENTDATASETPARQUET.parquet"
    negativeDatasetPath = "/home/fidel/Documents/NEGATIVESENTIMENTDATASETPARQUET.parquet"
    sentimentColName = "SentimentText"
    positiveColName = "words"
    negativeColName = "words"

    lexiconData = {
        pc.SENTIMENTDATASETPATH : reviewDatasetPath,
        pc.POSITIVEDATASETPATH : positiveDatasetPath,
        pc.NEGATIVEDATASETPATH : negativeDatasetPath,
        pc.SENTIMENTCOLNAME : sentimentColName,
        pc.POSITIVECOLNAME: positiveColName,
        pc.NEGATIVECOLNAME: negativeColName,
        pc.SPARK : sparkTest
    }

    lexiconSentiment = LexiconSentimentAnalysis()
    lexObj = lexiconSentiment.sentimentAnalysis(lexiconData)