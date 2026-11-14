# pyright: reportMissingImports=false
import numpy as np
import re
import VoiceClipProcessor as VCP
import GaussianTrainer as GT
from datetime import datetime

emotionList = GT.emotionList # ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']
emotionIntensityList = GT.emotionIntensityList # emotionIntensityList = ['neutralLow', 'calmLow', 'calmHigh', 'happyLow', 'happyHigh', 'sadLow', 'sadHigh', 'angryLow', 'angryHigh',
                                               #                         'fearfulLow', 'fearfulHigh', 'disgustLow', 'disgustHigh', 'surpriseLow', 'surpriseHigh']
emotionsToUse = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']


def testGMMAccuracy():
    print("Started training GMMs.")
    GT.rng = np.random.default_rng(GT.rngSeed)
    global emotionsToUse
    if GT.separateIntensities:
        emotionsToUse = emotionIntensityList
    else:
        emotionsToUse = emotionList
    if GT.usingPCA:
        GT.calculatePCA()
    for emotion in emotionsToUse:
        print(f"Started training GMMs for {emotion}.")
        GT.trainGMMsBestNumGaussians(emotion) #train the GMMs on exclusively training data
        print(f"Finished training GMMs for {emotion}.")
    accuracyPercent, ncpe, numGaussiansPerEmotion, emotionConfusions, currentIntensityCorrectNums = GT.evaluateGMMs(GT.actorsForTesting, None, False, printProgress=True)
    return accuracyPercent, ncpe, numGaussiansPerEmotion, emotionConfusions, currentIntensityCorrectNums

finalAccuracies = []
numsCorrectPerEmotion = []
numsGaussiansPerEmotion = []
confusionNumsPerEmotion = []
intensityCorrectNums = [] #not tied with GT.separateIntensities

trialNames = []
def runTrial(trialName):
    trialNum = len(trialNames)
    VCP.allActorNormalizationValues = {}
    VCP.allGenderActorStds = {0: [], 1: []} # 0 = female, 1 = male
    VCP.allActorIndivTotalNormVals = {}
    if VCP.numParamsChanged:
        GT.allRawPointValues = {}
        for a in range(1, 25):
            GT.allRawPointValues[a] = {}
            for c in range(1, 61):
                GT.allRawPointValues[a][c] = []
        VCP.numParamsChanged = False
    numsCorrectPerEmotion.append([])
    intensityCorrectNums.append({"Low":[0,0], "High":[0,0]}) #correct, total
    confusionNumsPerEmotion.append({'neutral':{}, 'calm':{}, 'happy':{}, 'sad':{}, 'angry':{}, 'fearful':{}, 'disgust':{}, 'surprise':{}})
    trialNames.append(trialName)
    GT.effectiveNumParams = GT.numPCADimensions if GT.usingPCA else VCP.numParams
   
    print(f"Started trial {trialNames.index(trialName)+1}: {trialName}.")
    accuracyPercent, ncpe, numGaussiansPerEmotion, emotionConfusions, currentIntensityCorrectNums = testGMMAccuracy()
    finalAccuracies.append(accuracyPercent)
    numsGaussiansPerEmotion.append(numGaussiansPerEmotion)
    for _, v in ncpe.items():
        numsCorrectPerEmotion[trialNum].append([v[0], v[1]])
    confusionNumsPerEmotion[trialNum] = emotionConfusions
    intensityCorrectNums[trialNum] = currentIntensityCorrectNums
       
    GT.DataPoint.allDataPoints = {emotion: [] for emotion in GT.DataPoint.allDataPoints} #necessary
    GT.Gaussian.allGaussians = {emotion: [] for emotion in GT.Gaussian.allGaussians}
    GT.KMean.allKMeans = {emotion: [] for emotion in GT.KMean.allKMeans} #just in case :P
    GT.PCA = None
    GT.Scaler = None
    GT.valAccComplete = False
    GT.validationDataPoints.clear()
    GT.allValidationDataPoints.clear()
    GT.validationPointCoords = None
    for e in emotionList:
        GT.allPointCoords[e] = None
    GT.optimalNumsGaussians = {'neutral':0, 'calm':0, 'happy':0, 'sad':0, 'angry':0, 'fearful':0, 'disgust':0, 'surprise':0}
    if usingLargeTrials: print(f"Current trial group name: {trialNamesLarge[-1]}")
    print(f"Actors used for testing: {GT.actorsForTesting}")
    print(f"Final accuracy: {finalAccuracies[trialNum]:.2f}%")
    print(f"Finished trial {trialNames.index(trialName)+1}: {trialName}.")
    return trialNum

usingLargeTrials = False
finalAccuraciesLarge = []
numsCorrectPerEmotionLarge = []
trialNamesLarge = []
largeTrialTrialNums = []
numsGaussiansPerEmotionAvgLarge = []
confusionNumsPerEmotionLarge = []
intensityCorrectNumsLarge = [] #not related to GT.separateIntensities


def runLargeTrial(trialName):
    global usingLargeTrials
    global confusionNumsPerEmotionLarge
    usingLargeTrials = True
    largeTrialNum = len(trialNamesLarge)
    numsCorrectPerEmotionLarge.append([])
    largeTrialTrialNums.append([])
    numsGaussiansPerEmotionAvgLarge.append({'neutral':0, 'calm':0, 'happy':0, 'sad':0, 'angry':0, 'fearful':0, 'disgust':0, 'surprise':0})
    confusionNumsPerEmotionLarge.append({'neutral':{}, 'calm':{}, 'happy':{}, 'sad':{}, 'angry':{}, 'fearful':{}, 'disgust':{}, 'surprise':{}})
    for e in confusionNumsPerEmotionLarge[-1]:
        confusionNumsPerEmotionLarge[-1][e] = {'neutral':0, 'calm':0, 'happy':0, 'sad':0, 'angry':0, 'fearful':0, 'disgust':0, 'surprise':0}
    intensityCorrectNumsLarge.append({"Low":[0,0], "High":[0,0]}) #correct, total
    trialNamesLarge.append(trialName)
    trialNums = []
    print(f"Started trial group {trialNamesLarge.index(trialName)+1}: {trialName}.")
   
    GT.actorsForTraining = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    GT.actorsForValidation = [22,23,24]
    GT.actorsForTesting = [1,2,3,4,5,6]
    trialNums.append(runTrial("testing actors: 1-6"))
    GT.actorsForTraining = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    GT.actorsForValidation = [1,2,3]
    GT.actorsForTesting = [4,5,6,7,8,9]
    trialNums.append(runTrial("testing actors: 4-9"))
    GT.actorsForTraining = [1,2,3,13,14,15,16,17,18,19,20,21,22,23,24]
    GT.actorsForValidation = [4,5,6]
    GT.actorsForTesting = [7,8,9,10,11,12]
    trialNums.append(runTrial("testing actors: 7-12"))
    GT.actorsForTraining = [1,2,3,4,5,6,16,17,18,19,20,21,22,23,24]
    GT.actorsForValidation = [7,8,9]
    GT.actorsForTesting = [10,11,12,13,14,15]
    trialNums.append(runTrial("testing actors: 10-15"))
    GT.actorsForTraining = [1,2,3,4,5,6,7,8,9,19,20,21,22,23,24]
    GT.actorsForValidation = [10,11,12]
    GT.actorsForTesting = [13,14,15,16,17,18]
    trialNums.append(runTrial("testing actors: 13-18"))
    GT.actorsForTraining = [1,2,3,4,5,6,7,8,9,10,11,12,22,23,24]
    GT.actorsForValidation = [13,14,15]
    GT.actorsForTesting = [16,17,18,19,20,21]
    trialNums.append(runTrial("testing actors: 16-21"))
    GT.actorsForTraining = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    GT.actorsForValidation = [16,17,18]
    GT.actorsForTesting = [19,20,21,22,23,24]
    trialNums.append(runTrial("testing actors: 19-24"))
    GT.actorsForTraining = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    GT.actorsForValidation = [19,20,21]
    GT.actorsForTesting = [22,23,24,1,2,3]
    trialNums.append(runTrial("testing actors: 22-24 & 1-3"))
       
    largeTrialTrialNums[largeTrialNum] = trialNums
    finalAccuraciesLarge.append(np.mean([finalAccuracies[t] for t in trialNums]))
    totalNumsGaussiansPerEmotion = {'neutral':0, 'calm':0, 'happy':0, 'sad':0, 'angry':0, 'fearful':0, 'disgust':0, 'surprise':0}
    for t in trialNums:
        numsCorrectPerEmotionLarge[largeTrialNum].append(numsCorrectPerEmotion[t])
        for e in emotionList:
            totalNumsGaussiansPerEmotion[e] += numsGaussiansPerEmotion[t][e]
            for e2 in emotionList:
                confusionNumsPerEmotionLarge[largeTrialNum][e][e2] += confusionNumsPerEmotion[t][e][e2]
        for k in intensityCorrectNums[t].keys():
            for v in range(len(intensityCorrectNums[t].values())):
                intensityCorrectNumsLarge[largeTrialNum][k][v] += intensityCorrectNums[t][k][v]
    for em, v in totalNumsGaussiansPerEmotion.items():
        numsGaussiansPerEmotionAvgLarge[largeTrialNum][em] = float(v) / float(len(trialNums))
    
    print(f"Trial group final accuracy: {finalAccuraciesLarge[largeTrialNum]:.2f}%")
    print(f"Finished trial group {trialNamesLarge.index(trialName)+1}: {trialName}.")


def excludeSpecific(paramsToExclude):
    VCP.setParametersToExclude(baseExclusions + paramsToExclude)

baseExclusions = ["spec_centroid_std", "spec_rolloff_std", "f0_range", "f0_std", "f0_slope_firstHalf", "f0_slope_secondHalf", "spec_centroid_mean", "speechRate", "spec_rolloff_mean"]
excludeSpecific([])
GT.covarianceType = 'diag'
GT.togglePCA(False)
GT.setScoringStyle = 'validationLL'
VCP.setNumMFCCs(13)
VCP.setStdCalculationStyle('individual_total')
VCP.setActorNeutralClipStdWeight(0.589)
VCP.setNumSegments(3)
GT.setRngSeed(185)

startTime = datetime.now()

runLargeTrial(f"final best test")

endTime = datetime.now()
executionTime = endTime - startTime


print("\n")
if usingLargeTrials:
    print("Final Results:")
    for largeTrialIndex, largeTrialName in enumerate(trialNamesLarge):
        print(f"Trial group {largeTrialIndex + 1}: {largeTrialName}")
        totalCorrectPerEmotion = [0 for ei in range(len(emotionList))]
        totalGuessesPerEmotion = [0 for ei in range(len(emotionList))]
        #print("Number correct for each emotion in constituent trials: ")
        for t in range(len(numsCorrectPerEmotionLarge[largeTrialIndex])):
            print(f"Trial {t+1}:", end=" ")
            #print("Number correct for each emotion: ", end="")
            ncpe = numsCorrectPerEmotionLarge[largeTrialIndex][t]
            for ei in range(len(emotionList)):
                v = ncpe[ei]
                totalCorrectPerEmotion[ei] += v[0]
                totalGuessesPerEmotion[ei] += v[1]
            #    if v[1] > 0:
            #        print(f"{emotionList[ei]}: {v[0]}/{v[1]} ({(100*v[0]/v[1]):.2f}%), ", end="")
            #    else:
            #        print(f"{emotionList[ei]}: {v[0]}/{v[1]} (N/A), ", end="")
            #        print(f"\nWARNING: No testing examples for {emotionList[ei]}.")
            #print("\n")
            currentFinalAcc = finalAccuracies[largeTrialTrialNums[largeTrialIndex][t]]
            print(f"Final accuracy: {currentFinalAcc:.2f}%")
            #for e in emotionList:
            #    print(f"{e}: {numsGaussiansPerEmotion[tr][e]},  ", end="")
        print("Overall number correct for each emotion: ", end="")
        for ei in range(len(emotionList)):
            numCorrect = totalCorrectPerEmotion[ei]
            numGuesses = totalGuessesPerEmotion[ei]
            if numGuesses > 0:
                print(f"{emotionList[ei]}: {numCorrect}/{numGuesses} ({(100*numCorrect/numGuesses):.2f}%), ", end="")
            else:
                print(f"{emotionList[ei]}: {numCorrect}/{numGuesses} (N/A), ", end="")
                print(f"\nWARNING: number of guesses was zero for {emotionList[ei]}. ", end="")
        print("Overall confusions for each emotion: ")
        for ei, e in enumerate(emotionList):
            totalIncorrectCurrentEmotion = totalGuessesPerEmotion[ei] - totalCorrectPerEmotion[ei]
            if totalIncorrectCurrentEmotion == 0:
                print(f"{e} - No incorrect guesses")
                continue
            confusionFound = False
            confusions = {}
            for e2 in emotionList:
                numTimesGuessed = confusionNumsPerEmotionLarge[largeTrialIndex][e][e2]
                confusionPercent = numTimesGuessed / totalIncorrectCurrentEmotion
                if confusionPercent >= 0.1:
                    confusions[e2] = confusionPercent
                    confusionFound = True
            print(f"{e} - {', '.join(f'{e2}: {(confusions[e2]*100):.2f}%' for e2 in confusions)}", end="")
            if confusionFound:
                print(f", other: {(100*(1 - sum([p for p in confusions.values()]))):.2f}%")
            else:
                print("no confusions over 10% were found.")
        print("Average number of gaussians per emotion: ", end="")
        for e in emotionList:
            print(f"{e}: {numsGaussiansPerEmotionAvgLarge[largeTrialIndex][e]},  ", end="")
        print("\n")
        print(f"Low intensity final accuracy: {(100*(intensityCorrectNumsLarge[largeTrialIndex]["Low"][0] / intensityCorrectNumsLarge[largeTrialIndex]["Low"][1])):.2f}%") #don't have to worry about dividing by zero here
        print(f"High intensity final accuracy: {(100*(intensityCorrectNumsLarge[largeTrialIndex]["High"][0] / intensityCorrectNumsLarge[largeTrialIndex]["High"][1])):.2f}%")
        print(f"Overall final accuracy: {finalAccuraciesLarge[largeTrialIndex]:.2f}%")
    print("\n")
    print("Summary:")
    for largeTrialIndex, largeTrialName in enumerate(trialNamesLarge):
        print(f"Trial group {largeTrialIndex + 1}: {largeTrialName} - Final accuracy: {finalAccuraciesLarge[largeTrialIndex]:.2f}%")
    bestIndex = finalAccuraciesLarge.index(max(finalAccuraciesLarge))
    print(f"Best trial: Trial {bestIndex+1}: {trialNamesLarge[bestIndex]} - Final accuracy: {finalAccuraciesLarge[bestIndex]:.2f}%")
        
else: #not updated to print all analysis types (e.g. confusions, per-intensity final accuracies)
    for trialIndex, trialName in enumerate(trialNames):
        print(f"Trial {trialIndex+1}: {trialName}")
        print("Number correct for each emotion: ", end="")
        ncpe = numsCorrectPerEmotion[trialIndex]
        for ei in range(len(emotionList)):
            v = ncpe[ei]
            print(f"{emotionList[ei]}: {v[0]}/{v[1]} ({(100*v[0]/v[1]):.2f}%), ", end="")
        print("\n")
        print(f"\nFinal accuracy: {finalAccuracies[trialIndex]:.2f}%")
        print("\n")

#print(f"Start time: {startTime}")
print(f"End time: {endTime}")
print(f"Execution time: {executionTime}")