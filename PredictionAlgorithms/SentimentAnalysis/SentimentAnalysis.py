from abc import ABC, abstractmethod
from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities


class SentimentAnalysis(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sentimentAnalysis(self, sentimentInfoData):
        raise NotImplementedError("subClass must implement abstract method")

    def textPreProcessing(self, sentimentInfoData):
        sentimentColName = sentimentInfoData.get(pc.SENTIMENTCOLNAME)
        dataset = sentimentInfoData.get(pc.DATASET)

        textProcessing = TextProcessing()
        '''add internal Id to the dataset'''
        dataset = textProcessing.toStringDatatype(dataset, sentimentColName)
        dataset = textProcessing.replaceSpecialChar(dataset, sentimentColName)
        dataset = textProcessing.createToken(dataset, sentimentColName)
        dataset = textProcessing.stopWordsRemover(dataset, pc.DMXTOKENIZED)
        dataset = textProcessing.lemmatization(dataset, pc.DMXSTOPWORDS)

        return dataset
