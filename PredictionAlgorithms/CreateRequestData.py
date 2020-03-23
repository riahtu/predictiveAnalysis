from PredictionAlgorithms.PredictiveConstants import PredictiveConstants


class CreateRequestData():
    predictiveData = {}

    def __init__(self):
        pass

    def createFeatureSelectionData(self, requestData):
        fileLocation = requestData.get(PredictiveConstants.FILELOCATION)
        feature_colm_req = requestData.get(PredictiveConstants.PREDICTOR)
        label_colm_req = requestData.get(PredictiveConstants.TARGET)
        algo_name = requestData.get(PredictiveConstants.ALGORITHMNAME)
        relation_list = requestData.get(PredictiveConstants.RELATIONSHIP_LIST)
        relation = requestData.get(PredictiveConstants.RELATIONSHIP)
        featureId = requestData.get(PredictiveConstants.FEATUREID)
        requestType = requestData.get(PredictiveConstants.REQUESTTYPE)
        locationAddress = requestData.get(PredictiveConstants.LOCATIONADDRESS)
        datasetName = requestData.get(PredictiveConstants.DATASETNAME)
        modelSheetName = requestData.get(PredictiveConstants.MODELSHEETNAME)

        self.predictiveData.update({
            PredictiveConstants.DATASETADD: fileLocation,
            PredictiveConstants.FEATURESCOLM: feature_colm_req,
            PredictiveConstants.LABELCOLM: label_colm_req,
            PredictiveConstants.ALGORITHMNAME: algo_name,
            PredictiveConstants.RELATIONSHIP_LIST: relation_list,
            PredictiveConstants.RELATIONSHIP: relation,
            PredictiveConstants.REQUESTTYPE: requestType,
            PredictiveConstants.LOCATIONADDRESS: locationAddress,
            PredictiveConstants.DATASETNAME: datasetName,
            PredictiveConstants.MODELID: featureId,
            PredictiveConstants.MODELSHEETNAME: modelSheetName,
        })
        return self.predictiveData

    def createRegressionModelData(self, requestData):
        fileLocation = requestData.get(PredictiveConstants.FILELOCATION)
        feature_colm_req = requestData.get(PredictiveConstants.PREDICTOR)
        label_colm_req = requestData.get(PredictiveConstants.TARGET)
        algo_name = requestData.get(PredictiveConstants.ALGORITHMNAME)
        relation_list = requestData.get(PredictiveConstants.RELATIONSHIP_LIST)
        relation = requestData.get(PredictiveConstants.RELATIONSHIP)
        trainDataLimit = requestData.get(PredictiveConstants.TRAINDATALIMIT)
        modelId = requestData.get(PredictiveConstants.MODELID)
        requestType = requestData.get(PredictiveConstants.REQUESTTYPE)
        locationAddress = requestData.get(PredictiveConstants.LOCATIONADDRESS)
        datasetName = requestData.get(PredictiveConstants.DATASETNAME)
        modelSheetName = requestData.get(PredictiveConstants.MODELSHEETNAME)
        modelSheetName = PredictiveConstants.PREDICTION_ + modelSheetName

        self.predictiveData.update({
            PredictiveConstants.DATASETADD: fileLocation,
            PredictiveConstants.FEATURESCOLM: feature_colm_req,
            PredictiveConstants.LABELCOLM: label_colm_req,
            PredictiveConstants.ALGORITHMNAME: algo_name,
            PredictiveConstants.RELATIONSHIP_LIST: relation_list,
            PredictiveConstants.RELATIONSHIP: relation,
            PredictiveConstants.TRAINDATALIMIT: trainDataLimit,
            PredictiveConstants.MODELID: modelId,
            PredictiveConstants.REQUESTTYPE: requestType,
            PredictiveConstants.LOCATIONADDRESS: locationAddress,
            PredictiveConstants.DATASETNAME: datasetName,
            PredictiveConstants.MODELSHEETNAME: modelSheetName,
        })
        return self.predictiveData

    def createPredictionData(self, requestData):
        fileLocation = requestData.get(PredictiveConstants.FILELOCATION)
        feature_colm_req = requestData.get(PredictiveConstants.PREDICTOR)
        label_colm_req = requestData.get(PredictiveConstants.TARGET)
        algo_name = requestData.get(PredictiveConstants.ALGORITHMNAME)
        relation_list = requestData.get(PredictiveConstants.RELATIONSHIP_LIST)
        relation = requestData.get(PredictiveConstants.RELATIONSHIP)
        modelId = requestData.get(PredictiveConstants.MODELID)
        requestType = requestData.get(PredictiveConstants.REQUESTTYPE)
        modelStorageLocation = requestData.get(PredictiveConstants.MODELSTORAGELOCATION)
        locationAddress = requestData.get(PredictiveConstants.LOCATIONADDRESS)
        datasetName = requestData.get(PredictiveConstants.DATASETNAME)
        modelSheetName = requestData.get(PredictiveConstants.MODELSHEETNAME)
        modelSheetName = PredictiveConstants.PREDICTION_ + modelSheetName

        self.predictiveData.update({
            PredictiveConstants.DATASETADD: fileLocation,
            PredictiveConstants.FEATURESCOLM: feature_colm_req,
            PredictiveConstants.LABELCOLM: label_colm_req,
            PredictiveConstants.ALGORITHMNAME: algo_name,
            PredictiveConstants.RELATIONSHIP_LIST: relation_list,
            PredictiveConstants.RELATIONSHIP: relation,
            PredictiveConstants.MODELID: modelId,
            PredictiveConstants.REQUESTTYPE: requestType,
            PredictiveConstants.LOCATIONADDRESS: locationAddress,
            PredictiveConstants.DATASETNAME: datasetName,
            PredictiveConstants.MODELSHEETNAME: modelSheetName,
            PredictiveConstants.MODELSTORAGELOCATION: modelStorageLocation
        })
        return self.predictiveData
