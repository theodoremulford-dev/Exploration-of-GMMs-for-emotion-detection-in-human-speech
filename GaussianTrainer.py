# pyright: reportMissingImports=false
import numpy as np
from scipy.special import logsumexp
import re
import VoiceClipProcessor as VCP
from sklearn.decomposition import PCA as PCAClass
from sklearn.preprocessing import StandardScaler


actorsForTraining = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
actorsForValidation = [16,17,18]
actorsForTesting = [19,20,21,22,23,24]
GMMTolerance, GMMMaxIter, GMMMinIter = 1e-9, 800, 400
KMeanTolerance, KMeanMaxIter = 1e-6, 50
maxKTestTrialNum = 10
covarianceRegularization = 1e-3
emotionList = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']
emotionIntensityList = ['neutralLow', 'calmLow', 'calmHigh', 'happyLow', 'happyHigh', 'sadLow', 'sadHigh', 'angryLow',
                        'angryHigh', 'fearfulLow', 'fearfulHigh', 'disgustLow', 'disgustHigh', 'surpriseLow', 'surpriseHigh']
setScoringStyle = 'validationLL' # 'validationAccuracy' for accuracy with the validation set, 'validationLL' for TLL with validation set, 'TLL' for highest total log likelyhood of the model itself
covarianceType = 'diag' # 'full' or 'diag'
gaussianCollapseThreshold = 5 # maximum number of data points a gaussian must have to be considered "collapsed" and cast aside
forceAllKValue = 0 # 0 for off, any number to force the K value for every emotion
complexityPenaltyMultiplier = 1 #used for BIC calculation
rngSeed = 0
separateIntensities = False #NOT SUPPORTED! WILL CRASH IN MANY PLACES!
usingPCA = False
numPCADimensions = 15
PCA = None
Scaler = None
effectiveNumParams = numPCADimensions if usingPCA else VCP.numParams

def setNumPCADimensions(newNumPCADimensions):
    global numPCADimensions
    if numPCADimensions != newNumPCADimensions:    
        numPCADimensions = newNumPCADimensions
        VCP.numParamsChanged = True

def togglePCA(newPCAState):
    global usingPCA
    if usingPCA != newPCAState:    
        usingPCA = newPCAState
        VCP.numParamsChanged = True

def setRngSeed(newRngSeed):
    global rngSeed
    if rngSeed != newRngSeed:    
        rngSeed = newRngSeed
        VCP.numParamsChanged = True


class Gaussian:
    allGaussians = {'neutral':[], 'calm':[], 'happy':[], 'sad':[], 'angry':[], 'fearful':[], 'disgust':[], 'surprise':[], 'neutralLow':[],
                    'calmLow':[], 'calmHigh':[], 'happyLow':[], 'happyHigh':[], 'sadLow':[], 'sadHigh':[], 'angryLow':[],
                    'angryHigh':[], 'fearfulLow':[], 'fearfulHigh':[], 'disgustLow':[], 'disgustHigh':[], 'surpriseLow':[], 'surpriseHigh':[]}
    optimalNumGaussians = {'neutral':0, 'calm':0, 'happy':0, 'sad':0, 'angry':0, 'fearful':0, 'disgust':0, 'surprise':0, 'neutralLow':0,
                           'calmLow':0, 'calmHigh':0, 'happyLow':0, 'happyHigh':0, 'sadLow':0, 'sadHigh':0, 'angryLow':0,
                           'angryHigh':0, 'fearfulLow':0, 'fearfulHigh':0, 'disgustLow':0, 'disgustHigh':0, 'surpriseLow':0, 'surpriseHigh':0}
    def __init__(self, weight, emotion):
        self._means = np.zeros(effectiveNumParams, dtype=float)
        self._covariances = np.eye(effectiveNumParams)
        self._weight = weight
        self._emotion = emotion
        Gaussian.allGaussians[emotion].append(self)
    @property
    def means(self):
        return self._means
    @property
    def covariances(self):
        return self._covariances
    @property
    def weight(self):
        return self._weight
    @property
    def emotion(self):
        return self._emotion
    @means.setter
    def means(self, newMeans):
        self._means = np.asarray(newMeans, dtype=float)
    @covariances.setter
    def covariances(self, newCovariances):
        self._covariances = np.asarray(newCovariances, dtype=float)
    @weight.setter
    def weight(self, newWeight):
        self._weight = float(newWeight)
   
class DataPoint:
    allDataPoints = {'neutral':[], 'calm':[], 'happy':[], 'sad':[], 'angry':[], 'fearful':[], 'disgust':[], 'surprise':[], 'neutralLow':[],
                    'calmLow':[], 'calmHigh':[], 'happyLow':[], 'happyHigh':[], 'sadLow':[], 'sadHigh':[], 'angryLow':[],
                    'angryHigh':[], 'fearfulLow':[], 'fearfulHigh':[], 'disgustLow':[], 'disgustHigh':[], 'surpriseLow':[], 'surpriseHigh':[]}
    def __init__(self, pointCoords, audio_path, emotion, intensity, addToGlobal=True):
        self._pointCoords = np.asarray(VCP.dictValsToArray(pointCoords), dtype=float)
        self._responsibilities = np.array([])
        self.audio_path = audio_path
        self._emotion = emotion
        self._intensity = intensity
        if addToGlobal:
            DataPoint.allDataPoints[emotion].append(self)
    @property
    def pointCoords(self):
        return self._pointCoords
    @property
    def responsibilities(self):
        return self._responsibilities
    @property
    def audio_path(self):
        return self._audio_path
    @property
    def emotion(self):
        return self._emotion
    @property
    def intensity(self):
        return self._intensity
    @pointCoords.setter
    def pointCoords(self, newPointCoords):
        self._pointCoords = np.asarray(newPointCoords, dtype=float)
    @responsibilities.setter
    def responsibilities(self, newResponsibilities):
        self._responsibilities = np.asarray(newResponsibilities, dtype=float)
    @audio_path.setter
    def audio_path(self, new_audio_path):
        self._audio_path = new_audio_path
       
class KMean:
    allKMeans = {'neutral':[], 'calm':[], 'happy':[], 'sad':[], 'angry':[], 'fearful':[], 'disgust':[], 'surprise':[], 'neutralLow':[],
                'calmLow':[], 'calmHigh':[], 'happyLow':[], 'happyHigh':[], 'sadLow':[], 'sadHigh':[], 'angryLow':[],
                'angryHigh':[], 'fearfulLow':[], 'fearfulHigh':[], 'disgustLow':[], 'disgustHigh':[], 'surpriseLow':[], 'surpriseHigh':[]}
    def __init__(self, emotion):
        self._means = np.zeros(effectiveNumParams, dtype=float)
        self._dataPoints = []
        self._emotion = emotion
        KMean.allKMeans[emotion].append(self)
       
    @classmethod
    def assignmentStep(cls, emotion):
        for km in KMean.allKMeans[emotion]:
            km.dataPoints = []
        for d in DataPoint.allDataPoints[emotion]:
            allKMeanDistances = [np.linalg.norm(d.pointCoords - km.means) for km in KMean.allKMeans[emotion]]
            closestKMeanIndex = allKMeanDistances.index(min(allKMeanDistances))
            KMean.allKMeans[emotion][closestKMeanIndex].dataPoints.append(d)  
    @classmethod
    def updateStep(cls, emotion):
        for km in KMean.allKMeans[emotion]:
            if len(km.dataPoints)>0:
                km.means = np.mean([d.pointCoords for d in km.dataPoints], axis=0)
    @classmethod
    def checkForConvergence(cls, emotion, tolerance, currentMeans, prevMeans):
        movements = np.linalg.norm(currentMeans - prevMeans, axis=1)
        return np.max(movements) < tolerance
   
    @property
    def emotion(self):
        return self._emotion
    @property
    def means(self):
        return self._means
    @property
    def dataPoints(self):
        return self._dataPoints
    @means.setter
    def means(self, newMeans):
        self._means = np.asarray(newMeans, dtype=float)
    @dataPoints.setter
    def dataPoints(self, newDataPoints):
        self._dataPoints = newDataPoints
   
allRawPointValues = {}
def initializeDataPoints(actorsToInclude, emotion, addPointToGlobal=True):
    dataPointList = []
    for a in actorsToInclude:
        for c in range(1, 61):
            audio_path = f"{VCP.pathToAllAudios}Actor_{a:02d}\\03-01-{(1 if c <= 4 else (c+3)//8 + 1):02d}-{(1 if c <= 4 else (c-5)//4 % 2 + 1):02d}-{((c-1)//2 % 2 + 1):02d}-{((c-1) % 2 + 1):02d}-{a:02d}.wav"
            audioInfo = VCP.extractInfoFromPath(audio_path)
            if audioInfo["emotion"] != emotion:
                continue
            if allRawPointValues[a][c] != []:
                dp = DataPoint(allRawPointValues[a][c], audio_path, emotion, audioInfo["intensity"], addToGlobal = addPointToGlobal)
                dataPointList.append(dp)
            else:
                if separateIntensities:
                    if audioInfo['emotion'] == re.sub(r'(Low|High)$', '', emotion) and audioInfo['intensity'] == re.search(r'(Low|High)$', emotion).group():
                        audioParameters = VCP.getParametersFromClip(audio_path)
                        if audioParameters != "clip skipped.":
                            dp = DataPoint(audioParameters, audio_path, emotion, audioInfo["intensity"], addToGlobal = addPointToGlobal)
                            dataPointList.append(dp)
                            allRawPointValues[a][c] = audioParameters
                else:
                    if audioInfo['emotion'] == emotion:
                        audioParameters = VCP.getParametersFromClip(audio_path)
                        if audioParameters != "clip skipped.":
                            dp = DataPoint(audioParameters, audio_path, emotion, audioInfo["intensity"], addToGlobal = addPointToGlobal)
                            dataPointList.append(dp)
                            allRawPointValues[a][c] = audioParameters
                    else:
                        continue
        print(f"Finished initializing data for Actor {a} (emotion: {emotion}).")
    return dataPointList

valAccComplete = False
def trainGMMsBestNumGaussians(emotion):
    if setScoringStyle in ('TLL', 'validationLL'):
        getBestNumGaussians(emotion)
        getBestCandidateTrial(emotion)
    elif setScoringStyle == 'validationAccuracy':
        if valAccComplete:
            return
        for e in emotionList:
            getBestNumGaussians(e)
        getBestCandidateTrial(emotion=None)


optimalNumsGaussians = {'neutral':0, 'calm':0, 'happy':0, 'sad':0, 'angry':0, 'fearful':0, 'disgust':0, 'surprise':0}
def getBestNumGaussians(emotion):
    if optimalNumsGaussians[emotion] != 0:
        return optimalNumsGaussians[emotion]
    else:
        if not usingPCA:
            initializeDataPoints(actorsForTraining, emotion, addPointToGlobal = True)
        print(f"Started finding optimal number of gaussians for {emotion}.")
        optimalNumsGaussians[emotion] = findOptimalNumGaussians(emotion)
        print(f"Finished finding optimal number of gaussians for {emotion}.")


validationDataPoints = {}
allValidationDataPoints = []
def getBestCandidateTrial(emotion=None):
    allFinalTLLs = []
    allValidationLLs = []
    allValidationAccuracies = []
    allGaussians = []
    print(f"Started training 25 trial GMMs for {emotion}. Set scoring style: {setScoringStyle}")
    nonCollapsedTrialComplete = False
   
    if setScoringStyle == 'TLL':
        i = 0
        while i < 20 or nonCollapsedTrialComplete == False:
            if i > 100:
                raise RuntimeError("100 executions to create a trial without a collapsed gaussian reached. Something's probably not set right.")
            finalTLL, trainedGaussians = initializeAndTrainGaussians(optimalNumsGaussians[emotion], emotion)
            numDataPoints = len(DataPoint.allDataPoints[emotion])
            collapsed = any(g.weight * numDataPoints < gaussianCollapseThreshold for g in trainedGaussians)
            if collapsed:
                print(f"Trial {i+1} rejected due to collapsed component.")
                i += 1
                continue
            else: nonCollapsedTrialComplete = True
            allFinalTLLs.append(finalTLL)
            allGaussians.append(trainedGaussians)
            print(f"Trial {i+1}/15 complete.")
            if i > 15:
                print(f"Valid trial found after {i+1} attempts.")
            i += 1
        highestFinalTLLIndex = allFinalTLLs.index(max(allFinalTLLs))
        return allGaussians[highestFinalTLLIndex] #returns a list of the gaussians in the best GMM
   
    elif setScoringStyle == 'validationLL':
        if emotion in validationDataPoints:
            currValidationDataPoints = validationDataPoints[emotion]
        else:
            if usingPCA:
                prePCAValidationDatapoints = initializeDataPoints(actorsForValidation, emotion, addPointToGlobal=False)
                vstackedValidationDataPoints = np.vstack([d.pointCoords for d in prePCAValidationDatapoints])
                scaledValidationDatapoints = Scaler.transform(vstackedValidationDataPoints)
                currValidationDataPoints = PCA.transform(scaledValidationDatapoints) #would be named pcaScaledValidationDatapoints but named like this due to later use in the function (for "readability" ofc not bc im lazy)
                validationDataPoints[emotion] = currValidationDataPoints
            else:
                currValidationDataPoints = initializeDataPoints(actorsForValidation, emotion, addPointToGlobal=False)
                validationDataPoints[emotion] = currValidationDataPoints
        i = 0
        while i < 20 or nonCollapsedTrialComplete == False:
            if i > 100:
                raise RuntimeError("100 executions to create a trial without a collapsed gaussian reached---something's probably not set right.")
            finalTLL, trainedGaussians = initializeAndTrainGaussians(optimalNumsGaussians[emotion], emotion)
            numDataPoints = len(DataPoint.allDataPoints[emotion])
            collapsed = any(g.weight * numDataPoints < gaussianCollapseThreshold for g in trainedGaussians)
            if collapsed:
                print(f"Trial {i+1} rejected due to collapsed component.")
                i += 1
                continue
            else: nonCollapsedTrialComplete = True
            validationLL = calculateValTLL(currValidationDataPoints, trainedGaussians)
            allValidationLLs.append(validationLL)
            allGaussians.append(trainedGaussians)
            if i > 20:
                print(f"Valid trial found after {i+1} attempts.")
            i += 1
        bestValTLLIndex = np.argmax(allValidationLLs)
        print(f"Finished training 20 trial GMMs for {emotion}.")
        return allGaussians[bestValTLLIndex]
 
    elif setScoringStyle == 'validationAccuracy':
        allValidationAccuracies = []
        currValidationDataPoints = []
        global allValidationDataPoints
        if allValidationDataPoints:
            currValidationDataPoints = allValidationDataPoints
        else:
            if usingPCA:
                prePCAValidationDatapoints = []
                for e in emotionList:
                    prePCAValidationDatapoints.extend(initializeDataPoints(actorsForValidation, e, addPointToGlobal=False))
                vstackedValidationDataPoints = np.vstack([d.pointCoords for d in np.ravel(prePCAValidationDatapoints)])
                scaledValidationDatapoints = Scaler.transform(vstackedValidationDataPoints)
                pcaValidationDataPoints = PCA.transform(scaledValidationDatapoints) #would be named pcaScaledValidationDatapoints but named like this due to later use in the function (for "readability" ofc not bc im lazy)
                for d, coords in zip(prePCAValidationDatapoints, pcaValidationDataPoints):
                    d.pointCoords = coords
                currValidationDataPoints = prePCAValidationDatapoints #after their pointCoords have been adjusted to pca
                allValidationDataPoints = currValidationDataPoints
            else:
                for e in emotionList:
                    currValidationDataPoints.extend(initializeDataPoints(actorsForValidation, e, addPointToGlobal=False))
                allValidationDataPoints = currValidationDataPoints
        for i in range(15):
            trialTrainedGaussians = {'neutral':[], 'calm':[], 'happy':[], 'sad':[], 'angry':[], 'fearful':[], 'disgust':[], 'surprise':[]}
            for e in emotionList:
                for attempt in range(40):
                    finalTLL, trainedGaussians = initializeAndTrainGaussians(optimalNumsGaussians[e], e)
                    numDataPoints = len(DataPoint.allDataPoints[e])
                    collapsed = any(g.weight * numDataPoints < gaussianCollapseThreshold for g in trainedGaussians)
                    if collapsed:
                        print(f"Attempt {attempt+1} rejected due to collapsed component.")
                        continue
                    else:
                        trialTrainedGaussians[e] = list(trainedGaussians)
                        break
                else:
                    raise RuntimeError(f"40 executions to create a trial without a collapsed gaussian reached for {e}---something's probably not set right.")
            if any(not trialTrainedGaussians[e] for e in emotionList): #check just incase the missing component is not detected during creation
                raise RuntimeError("At least one emotional model is missing from candidate set.")
            validationAccuracy, _, _, _, _ = evaluateGMMs(None, currValidationDataPoints, needPCAScaling=False, printProgress=False, gaussians=trialTrainedGaussians)
            allValidationAccuracies.append(validationAccuracy)
            allGaussians.append(trialTrainedGaussians)
        bestValAccIndex = np.argmax(allValidationAccuracies)
        print(f"Finished training 15 trial GMMs.")
        for e in emotionList:
            Gaussian.allGaussians[e] = allGaussians[bestValAccIndex][e]
        global valAccComplete
        valAccComplete = True
        return allGaussians[bestValAccIndex]
   
    else: raise RuntimeError("The value of setScoringStyle was not recognized as an implemented style.")

validationPointCoords = None
def calculateValTLL(dataPoints, gaussians):
    valTLL = 0.0
    global validationPointCoords
    if validationPointCoords is not None:
        allPointCoords = validationPointCoords
    else:
        if usingPCA:
            allPointCoords = np.asarray([d for d in dataPoints])
        else:
            allPointCoords = np.asarray([d.pointCoords for d in dataPoints])
        validationPointCoords = allPointCoords
    pointLogWeightedResps = np.column_stack([np.log(g.weight) + logProbDensFuncBatch(allPointCoords, g.means, g.covariances) for g in gaussians])
    allWeightedResps = logsumexp(pointLogWeightedResps, axis=1)
    valTLL = np.sum(allWeightedResps)
    return valTLL

numCorrectPerEmotion = {'neutral':[0,0], 'calm':[0,0], 'happy':[0,0], 'sad':[0,0], 'angry':[0,0], 'fearful':[0,0], 'disgust':[0,0], 'surprise':[0,0]} # [correct, total]
def evaluateGMMs(actorsToEvaluate, pointsToEvaluate, needPCAScaling, printProgress=False, gaussians=None):
    if gaussians is None:
        gaussians = Gaussian.allGaussians
    numGaussiansPerEmotion = {'neutral':0, 'calm':0, 'happy':0, 'sad':0, 'angry':0, 'fearful':0, 'disgust':0, 'surprise':0}
    for e in emotionList:
        numGaussiansPerEmotion[e] += len(gaussians[e])
    if pointsToEvaluate is None:
        if printProgress: print("Started initializing evaluation data.")
        dataPointList = []
        for emotion in emotionList:
            dataPointList.extend(initializeDataPoints(actorsToEvaluate, emotion, addPointToGlobal=False))
            if printProgress: print(f"Finished initializing evaluation data for {emotion}.")
        if printProgress: print(f"Finished initializing all evaluation data.")
    else: dataPointList = pointsToEvaluate
    if needPCAScaling or pointsToEvaluate is None:
        if usingPCA:
            vstackedTestingDataPoints = np.vstack([d.pointCoords for d in dataPointList])
            scaledTestingDatapoints = Scaler.transform(vstackedTestingDataPoints)
            pcaScaledTestingDatapoints = PCA.transform(scaledTestingDatapoints)
            for d, coords in zip(dataPointList, pcaScaledTestingDatapoints):
                d.pointCoords = coords
            if printProgress: print("Finished PCA transforming all evaluation data.")
            
    if printProgress: print("Started evaluating GMMs")
    global numCorrectPerEmotion
    numCorrectPerEmotion = {'neutral':[0,0], 'calm':[0,0], 'happy':[0,0], 'sad':[0,0], 'angry':[0,0], 'fearful':[0,0], 'disgust':[0,0], 'surprise':[0,0]}
    emotionConfusions = {'neutral':{}, 'calm':{}, 'happy':{}, 'sad':{}, 'angry':{}, 'fearful':{}, 'disgust':{}, 'surprise':{}}
    for e in emotionConfusions:
        emotionConfusions[e] = {'neutral':0, 'calm':0, 'happy':0, 'sad':0, 'angry':0, 'fearful':0, 'disgust':0, 'surprise':0}
    intensityCorrectNums = {"Low":[0,0], "High":[0,0]}
    numCorrectGuesses = 0
    numDataPoints = len(dataPointList)
    dataPointNumber = 0
    for d in dataPointList:
        dataPointNumber += 1
        if separateIntensities:
            emotionLikelyhoods = {'neutralLow':0.0, 'calmLow':0.0, 'calmHigh':0.0, 'happyLow':0.0, 'happyHigh':0.0, 'sadLow':0.0, 'sadHigh':0.0, 'angryLow':0.0,
                                  'angryHigh':0.0, 'fearfulLow':0.0, 'fearfulHigh':0.0, 'disgustLow':0.0, 'disgustHigh':0.0, 'surpriseLow':0.0, 'surpriseHigh':0.0}
            wholeEmotionLikelyHoods = {'neutral':0.0, 'calm':0.0, 'happy':0.0, 'sad':0.0, 'angry':0.0, 'fearful':0.0, 'disgust':0.0, 'surprise':0.0}
        else:
            emotionLikelyhoods = {'neutral':0.0, 'calm':0.0, 'happy':0.0, 'sad':0.0, 'angry':0.0, 'fearful':0.0, 'disgust':0.0, 'surprise':0.0}
        for emotion in emotionList:
            pointLogWeightedProbs = [np.log(g.weight) + logProbDensFunc(d.pointCoords, g.means, g.covariances) for g in gaussians[emotion]]
            emotionLogLikelyhood = logsumexp(pointLogWeightedProbs)
            emotionLikelyhoods[emotion] = emotionLogLikelyhood
        if separateIntensities:
            for emotion in emotionList:
                if emotion == "neutral":
                    wholeEmotionLikelyHoods[emotion] = emotionLikelyhoods["neutralLow"]
                else:
                    wholeEmotionLikelyHoods[emotion] = logsumexp([
                        emotionLikelyhoods[f'{emotion}High'] + np.log(0.5),
                        emotionLikelyhoods[f'{emotion}Low'] + np.log(0.5)
                    ])
            emotionGuess = max(wholeEmotionLikelyHoods, key=wholeEmotionLikelyHoods.get)
        else:          
            emotionGuess = max(emotionLikelyhoods, key=emotionLikelyhoods.get)
        if emotionGuess == re.sub(r'(Low|High)$', '', d.emotion):
            numCorrectGuesses += 1
            numCorrectPerEmotion[emotionGuess][0] += 1
            intensityCorrectNums[d.intensity][0] += 1
            if printProgress: print(f"Evaluating GMMs ; progress: {(100*dataPointNumber/numDataPoints):.2f}% ; Guessed correctly! Guess: {emotionGuess}, Correct: {d.emotion}")
        else:
            emotionConfusions[d.emotion][emotionGuess] += 1
            if printProgress: print(f"Evaluating GMMs ; progress: {(100*dataPointNumber/numDataPoints):.2f}% ; Guessed incorrectly. Guess: {emotionGuess}, Correct: {d.emotion}")
        numCorrectPerEmotion[re.sub(r'(Low|High)$', '', d.emotion)][1] += 1
        intensityCorrectNums[d.intensity][1] += 1
           
    return (numCorrectGuesses / len(dataPointList)) * 100, numCorrectPerEmotion, numGaussiansPerEmotion, emotionConfusions, intensityCorrectNums #percentage correct, numCorrectPerEmotion[correct, guesses], num guassinas per emotion

def findOptimalNumGaussians(emotion):
    validationDataPoints.clear()
    allBestBICs = []
    if forceAllKValue:
        return forceAllKValue
    else:
        for K in range(1, 6): #numbers of gaussians to try
            allBICs = []
            print(f"Trying K value {K}.")
            trial = 0
            numNonCollapsedtrials = 0
            while numNonCollapsedtrials < 8 and trial < maxKTestTrialNum: #multiple trials to prevent unlucky initialization
                finalTLL, trainedGaussians = initializeAndTrainGaussians(K, emotion)
                numDataPoints = len(DataPoint.allDataPoints[emotion])
                collapsed = any(g.weight * numDataPoints < gaussianCollapseThreshold for g in trainedGaussians)
                if collapsed:
                    print(f"Trial {trial+1} rejected due to collapsed component.")
                    if trial == maxKTestTrialNum - 1:
                        allBICs.append(np.inf)
                        print(f"No non-collapsed model of {emotion} found with K value {K}.")
                        break
                    trial += 1
                    continue
                numNonCollapsedtrials += 1
                allBICs.append(calculateBIC(emotion, finalTLL))
                print(f"Trial {trial + 1} complete.")
                trial += 1
            allBestBICs.append(min(allBICs))
        optimalNumGaussians = 1 + allBestBICs.index(min(allBestBICs))
        Gaussian.optimalNumGaussians[emotion] = optimalNumGaussians
        return optimalNumGaussians

def initializeAndTrainGaussians(numGaussians, emotion):
    Gaussian.allGaussians[emotion] = [] #clear to make indexing work properly
    KMean.allKMeans[emotion] = []
    createdGaussians = initializeGaussiansSmart(numGaussians, emotion)
    return [runGMMTrainingLoop(emotion), createdGaussians] #returns the final total log likelyhood for that GMM

def initializeGaussiansSmart(numGaussians, emotion):
    if numGaussians == 1: #get the starting values directly from the mean/cov of the entire dataset for that emotion
        print("numGaussians is 1, so skipping k-means.")
        currentEmotionDataset = np.array([d.pointCoords for d in DataPoint.allDataPoints[emotion]])
        g = Gaussian(1.0, emotion)
        g.means = np.mean(currentEmotionDataset, axis=0)
        if covarianceType == 'full': g.covariances = np.cov(currentEmotionDataset, rowvar=False) + covarianceRegularization * np.eye(len(DataPoint.allDataPoints[emotion][0].pointCoords))
        elif covarianceType == 'diag': g.covariances = np.diag(np.diag(np.cov(currentEmotionDataset, rowvar=False))) + covarianceRegularization * np.eye(len(DataPoint.allDataPoints[emotion][0].pointCoords))
        else: raise RuntimeError("The value of covarianceType was not recognized as an implemented covariance type.")
        return [g]
    print(f"Randomly initializing {numGaussians} k-means for {emotion}.")
    randomInitializeKMeans(numGaussians, emotion)
    runKMeanTrainingLoop(emotion)
    globalPointCoords = np.array([d.pointCoords for d in DataPoint.allDataPoints[emotion]])
    if covarianceType == 'full': globalCovariance = np.cov(globalPointCoords, rowvar=False) + covarianceRegularization * np.eye(globalPointCoords.shape[1])
    elif covarianceType == 'diag': globalCovariance = np.diag(np.diag(np.cov(globalPointCoords, rowvar=False))) + covarianceRegularization * np.eye(globalPointCoords.shape[1])
    else: raise RuntimeError("The value of covarianceType was not recognized as an implemented covariance type.")
    kMeanCovariances = []
    allPointCoords = np.array([d.pointCoords for d in DataPoint.allDataPoints[emotion]])
    #if len(allPointCoords) > 1:
    #    currentKMeanCov = (np.cov([d.pointCoords for d in km.dataPoints], rowvar=False) + covarianceRegularization * np.eye(allPointCoords.shape[1]))
    #else:
    #    currentKMeanCov = globalCovariance.copy()
    #kMeanCovariances.append(currentKMeanCov)
    kMeanCovariances = [globalCovariance.copy() for _ in KMean.allKMeans[emotion]]
    return initializeGaussiansFromMeans(numGaussians, emotion, [km.means for km in KMean.allKMeans[emotion]], kMeanCovariances) #returns an array of all gaussians created

rng = np.random.default_rng(rngSeed)
def randomInitializeKMeans(numKMeans, emotion):
    dataPointsToClone = rng.choice(DataPoint.allDataPoints[emotion], size=numKMeans, replace=False)
    for i in range(len(dataPointsToClone)):
        km = KMean(emotion)
        km.means = (dataPointsToClone[i].pointCoords).copy()


def runKMeanTrainingLoop(emotion):
    prevMeans = np.array([])
    numIter = 0
    converged = False
    while converged == False and numIter < KMeanMaxIter:
        KMean.assignmentStep(emotion)
        KMean.updateStep(emotion)
        currentMeans = np.array([km.means for km in KMean.allKMeans[emotion]])
        if numIter != 0:
            converged = KMean.checkForConvergence(emotion, KMeanTolerance, currentMeans, prevMeans)
            if converged == True:
                print(f"Finished running k-means training loop in {numIter} iterations.", end=" ")
                break
        prevMeans = currentMeans
        numIter += 1


def initializeGaussiansFromMeans(numGaussians, emotion, inputMeans, inputCovariances): #inputMeans being a 2d array of gaussian# x means
    gmmGaussians = []
    for i in range(numGaussians):
        g = Gaussian(1 / numGaussians, emotion)
        g.means = inputMeans[i]
        g.covariances = inputCovariances[i]
        gmmGaussians.append(g)
    return gmmGaussians
   
def runGMMTrainingLoop(emotion):
    print("Started running GMM training loop.")
    prevTLL = 0.0
    numIter = 0
    converged = False
    while (converged == False and numIter < GMMMaxIter) or numIter < GMMMinIter:
        tLL, allResponsibilities = ExpectationStep(emotion) #totalLogLikelyhood
        MaximizationStep(emotion, allResponsibilities)
        if abs(tLL - prevTLL) < GMMTolerance and numIter >= GMMMinIter:
            print(f"Finished running GMM training loop in {numIter} iterations.", end=" ")
            finalTLL, _ = ExpectationStep(emotion) #finalTotalLogLikelyhood
            converged = True
            return finalTLL
        if GMMMaxIter - numIter == 1: # checks if the loop is on its final iteration
            print("Finished running GMM training loop (reached max iteration limit).")
            finalTLL, _ = ExpectationStep(emotion)
            return finalTLL
        prevTLL = tLL
        numIter += 1
       
def calculatePCA():
    print("Started calculating PCA.")
    emotions = emotionIntensityList if separateIntensities else emotionList
    for emotion in emotions:
        initializeDataPoints(actorsForTraining, emotion, addPointToGlobal=True)
        print(f"Finished initializing training datapoints for {emotion}.")
    allTrainingDatapoints = np.vstack([d.pointCoords for emotion in emotions for d in DataPoint.allDataPoints[emotion]])
    scaler = StandardScaler()
    scaledTrainingDatapoints = scaler.fit_transform(allTrainingDatapoints)
    pca = PCAClass(n_components=numPCADimensions)
    pcaTransformedTrainingDatapoints = pca.fit_transform(scaledTrainingDatapoints)
   
    i = 0
    for emotion in emotions:
        for d in DataPoint.allDataPoints[emotion]:
            d.pointCoords = pcaTransformedTrainingDatapoints[i]
            i += 1
    global PCA
    PCA = pca
    global Scaler
    Scaler = scaler
    print("Finished calculating PCA and transforming datapoints.")
   
def calculateBIC(emotion, finalTLL): #Bayesian Information Criterion: scores a model based on its encapsulation of the data while penalizing overcomplexity
    numGaussians = len(Gaussian.allGaussians[emotion])
    numDataPoints = len(DataPoint.allDataPoints[emotion])
    numDimensions = len(DataPoint.allDataPoints[emotion][0].pointCoords)
   
    if covarianceType == 'full': numFreeParameters = numGaussians * numDimensions * (numDimensions + 3) / 2  + numGaussians - 1
    elif covarianceType == 'diag': numFreeParameters = 2 * numGaussians * numDimensions + numGaussians - 1
    else: raise RuntimeError("The value of covarianceType was not recognized as an implemented covariance type.")
    return complexityPenaltyMultiplier * numFreeParameters * np.log(numDataPoints) - 2 * finalTLL


allPointCoords = {'neutral':None, 'calm':None, 'happy':None, 'sad':None, 'angry':None, 'fearful':None, 'disgust':None, 'surprise':None}
def ExpectationStep(emotion):
    gaussians = Gaussian.allGaussians[emotion]
    global allPointCoords
    if allPointCoords[emotion] is not None:
        allCurrentPointCoords = allPointCoords[emotion]
    else:
        allCurrentPointCoords = np.asarray([d.pointCoords for d in DataPoint.allDataPoints[emotion]], dtype=float)
        allPointCoords[emotion] = allCurrentPointCoords
    pointLogWeightedResps = np.column_stack([np.log(g.weight) + logProbDensFuncBatch(allCurrentPointCoords, g.means, g.covariances) for g in gaussians])
    allWeightedResps = logsumexp(pointLogWeightedResps, axis=1)
    allResponsibilities = np.exp(pointLogWeightedResps - allWeightedResps[:, None])
    for i, d in enumerate(DataPoint.allDataPoints[emotion]):
        d.responsibilities = allResponsibilities[i]
    return np.sum(allWeightedResps), allResponsibilities #totalLogLikelyhood
           
def MaximizationStep(emotion, allResponsibilities):
    global allPointCoords
    if allPointCoords[emotion] is not None:
        allCurrentPointCoords = allPointCoords[emotion]
    else:
        allCurrentPointCoords = np.asarray([d.pointCoords for d in DataPoint.allDataPoints[emotion]], dtype=float) #[point, parameter]
        allPointCoords[emotion] = allCurrentPointCoords
    gTotalResps = np.sum(allResponsibilities, axis=0) #gTotalResps[g] = sum of resps to it of all datapoints
    numPoints, numParams = allCurrentPointCoords.shape
    for g in range(len(Gaussian.allGaussians[emotion])):
        cg = Gaussian.allGaussians[emotion][g] #current gaussian object
        cg.means = (allResponsibilities[:, g] @ allCurrentPointCoords) / gTotalResps[g]
        pointDiffs = allCurrentPointCoords - cg.means
        if covarianceType == 'full':
            cg.covariances = (pointDiffs.transpose() @ (allResponsibilities[:, g, None] * pointDiffs)) / gTotalResps[g]
        elif covarianceType == 'diag':
            cg.covariances = np.diag(np.sum(allResponsibilities[:, g, None] * (pointDiffs ** 2), axis=0) / gTotalResps[g])
        else: raise RuntimeError("The value of covarianceType was not recognized as an implemented covariance type.")
        cg.covariances += covarianceRegularization * np.eye(numParams) # prevents determinant becoming zero if it collapses on only a few data points
        cg.weight = float(gTotalResps[g] / numPoints)
    
def logProbDensFunc(pointCoords, means, covariances): #probabilityDistributionFunction for a multivariate gaussian distribution; single data point
    pointCoords = np.asarray(pointCoords, dtype=float)
    means = np.asarray(means, dtype=float)
    numParams = len(pointCoords)
    pointDiffs = pointCoords - means
    return(logProbDensFuncCalculation(pointDiffs, numParams, covariances, isBatch=False))

def logProbDensFuncBatch(allCurrentPointCoords, means, covariances): #multiple data points under the same gaussian
    allCurrentPointCoords = np.asarray(allCurrentPointCoords, dtype=float)
    means = np.asarray(means, dtype=float)
    numParams = allCurrentPointCoords.shape[1]
    pointDiffs = allCurrentPointCoords - means
    return(logProbDensFuncCalculation(pointDiffs, numParams, covariances, isBatch=True))
    
def logProbDensFuncCalculation(pointDiffs, numParams, covariances, isBatch):
    if not isBatch:
        pointDiffs = np.atleast_2d(pointDiffs)
    if covarianceType == 'full':
        covariances = np.atleast_2d(covariances)
        sign, logdetCovariances = np.linalg.slogdet(covariances)
        if sign < 0:
            raise np.linalg.LinAlgError("Covariance matrix had a negative determinant (must be positive).")
        elif sign == 0:
            raise np.linalg.LinAlgError("Covariance matrix had a determinant of zero (must be greater than zero).")
        results = (((-numParams / 2) * np.log(2 * np.pi) - .5 * logdetCovariances) + (-0.5 * np.sum(np.linalg.solve(covariances, pointDiffs.transpose()).transpose() * pointDiffs, axis=1))) # log-space
        #not log-space: return (1 / ((2 * np.pi) ** (numParams / 2) * np.sqrt(np.linalg.det(covariances)))) * np.exp(-0.5 * np.linalg.solve(covariances, pointDiffs) @ pointDiffs)
    elif covarianceType == 'diag':
        covariances = np.diag(covariances)
        logdetCovariances = np.sum(np.log(covariances))
        results = -0.5 * (numParams * np.log(2.0 * np.pi) + logdetCovariances + np.sum((pointDiffs ** 2) / covariances, axis=1))
    return results if isBatch else results[0]