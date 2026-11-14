# pyright: reportMissingImports=false
import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from parselmouth.praat import run_file
import tempfile
import shutil
import os


stdCalculationStyle = 'individual_total'  # 'individual'       = only calculated with that actor's neutral clips
                                          # 'gender'           = calculated with all neutral clips from actors of that gender
                                          # 'individual_total' = calculated with all of that actor's clips (neutral weighted higher)
currentActorStdWeight = 0.3       # -1 = normal calculation, decimal value = that percent of the final std used in calculation (0.5 = 50% weight)
actorNeutralClipStdWeight = .589    # -1 = normal calculation, decimal value = that percent of the final std used in calculation (0.5 = 50% weight)


numParams = 28
numMFCCs = 13
parametersToExclude = []
numParamsChanged = False
pathToAllAudios = 'C:\\Users\\Theod\\NOW-Sat\\.vscode\\sciencefair2026\\Audio_Speech_Actors_01-24\\'
numSegments = 3
frameLength = 2048 #in hz; ~93ms
hopLength = 512 # ~23ms

def extractInfoFromPath(audio_path):
        pathSegments = audio_path.strip().split("\\")
        actorNumber = int(pathSegments[-2][6:]) # 1-scaled
        emotion = ""
        emotionNumber = pathSegments[-1][6:8]
        match emotionNumber:
            case "01":
                emotion = "neutral"
            case "02":
                emotion = "calm"
            case "03":
                emotion = "happy"
            case "04":
                emotion = "sad"
            case "05":
                emotion = "angry"
            case "06":
                emotion = "fearful"
            case "07":
                emotion = "disgust"
            case "08":
                emotion = "surprise"
        intensity = ""
        match pathSegments[-1][9:11]:
            case "01":
                intensity = "Low"
            case "02":
                intensity = "High"
        clipNumber = 4*(int(pathSegments[-1][9:11])-1) + 2*(int(pathSegments[-1][12:14])-1) + 1*(int(pathSegments[-1][15:17])-1) # 4*intensity + 2*statement + 1*repetition; 0-scaled
        return {"actorNumber":actorNumber, "emotion":emotion, "emotionNumber":emotionNumber, "intensity":intensity, "clipNumber":clipNumber}

def getParametersFromClip(audio_path): #returns a dictionary of all an audio's parameters (z-scored)
    actorNumber = extractInfoFromPath(audio_path)["actorNumber"]
    extractedParameters = extractParameters(audio_path)
    if extractedParameters  == "clip skipped.":
        return "clip skipped."
    return normalizeParameters(audio_path, actorNumber, extractedParameters)

allClipValues = {} # {(audio_path, numSegments): extractedParameters}
def extractParameters(audio_path):
    cacheKey = (audio_path, numSegments)
    if cacheKey in allClipValues:
        allClipVals = allClipValues[cacheKey].copy()
        return applyParameterExclusions(allClipVals)
    
    #ALL PARAMETER SETUP
    minSearchFrequency, maxSearchFrequency = 40, 2088
    y, sr = librosa.load(audio_path)
    parsel_audio = parselmouth.Sound(audio_path)
   
    #F0
    f0, voiced_bool, voiced_probs = librosa.pyin(y, fmin=minSearchFrequency, fmax=maxSearchFrequency, sr=sr, frame_length=frameLength, hop_length=hopLength)
    valid = ~np.isnan(f0)
    pitches = f0[valid]
    if not np.any(~np.isnan(f0)):
        print("clip skipped.")
        return "clip skipped."
   
    #RMS (energy)
    rms = librosa.feature.rms(y=y, frame_length=frameLength, hop_length=hopLength)[0]
    rms_mean = np.mean(rms) #final param!
    rms_std = np.std(rms) #final param!
   
    #MFCCs
    mfccsPerWindow = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=numMFCCs, n_fft=frameLength, hop_length=hopLength)
    
    #Spectral Shape (timbre)
    stft = librosa.stft(y=y, n_fft=frameLength, hop_length=hopLength)
    magnitude = np.abs(stft)
    spec_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=.85)[0]
    
    #Final parameter extraction (split-segment)    
    frameTimes = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hopLength)
    clipDuration = len(y) / sr
    segmentedParameters = {}
    for segmentIndex in range(numSegments):
        segmentNumber = segmentIndex + 1
        segmentStart = clipDuration * segmentIndex / numSegments
        segmentEnd = clipDuration * (segmentIndex + 1) / numSegments
        segmentMask = ((frameTimes >= segmentStart) & (frameTimes < segmentEnd)) #gets all frames within the segment
        if not np.any(segmentMask):
            raise ValueError(f"No feature frames found in segment {segmentNumber} of {audio_path}. Look at segment mask calculation.")
        
        #F0
        segmentF0 = f0[segmentMask]
        segmentTimes = frameTimes[segmentMask]
        validPitchMask = ~np.isnan(segmentF0)
        pitches = segmentF0[validPitchMask]
        pitchTimes = segmentTimes[validPitchMask]
        
        if len(pitches) > 0:
            f0_mean = np.nanmean(pitches) #final param!
            f0_std = np.nanstd(pitches) #final param!
            f0_range = np.nanmax(pitches) - np.nanmin(pitches) #final param!
            if len(pitches) >=3:
                f0_slope, _ = np.polyfit(pitchTimes, pitches, 1) #final param!
            else:
                f0_slope = 0.0
        else: # No voiced frames in this segment
            f0_mean = 0.0
            f0_std = 0.0
            f0_range = 0.0
            f0_slope = 0.0
            
        f0_voiced_fraction = len(pitches) / len(segmentF0)
        
        #RMS (energy)
        segmentRms = rms[segmentMask]
        rms_mean = np.mean(segmentRms) #final param!
        rms_std = np.std(segmentRms) #final param!
        
        #MFCCs
        segmentMFCCs = mfccsPerWindow[:, segmentMask]
        mfccStds = np.std(segmentMFCCs, axis=1) #final param! x numMFCCs (probably 13)
        mfccMeans = np.mean(segmentMFCCs, axis=1) #final param! x numMFCCs (probably 13)
   
        #Spectral Shape (timbre)
        segmentCentroid = spec_centroid[segmentMask]
        spec_centroid_mean = np.mean(segmentCentroid) #final param!
        spec_centroid_std = np.std(segmentCentroid) #final param!
        
        segmentRolloff = spec_rolloff[segmentMask]
        spec_rolloff_mean = np.mean(segmentRolloff) #final param!
        spec_rolloff_std = np.std(segmentRolloff) #final param!
    
        prefix = f"segment{segmentNumber}_"
        segmentedParameters.update({
            prefix + "f0_mean": f0_mean,
            prefix + "f0_std": f0_std,
            prefix + "f0_range": f0_range,
            prefix + "f0_slope": f0_slope,
            prefix + "f0_voiced_fraction": f0_voiced_fraction,

            prefix + "rms_mean": rms_mean,
            prefix + "rms_std": rms_std,

            prefix + "mfccMeans": mfccMeans,
            prefix + "mfccStds": mfccStds,

            prefix + "spec_centroid_mean": spec_centroid_mean,
            prefix + "spec_centroid_std": spec_centroid_std,

            prefix + "spec_rolloff_mean": spec_rolloff_mean,
            prefix + "spec_rolloff_std": spec_rolloff_std})
    
    #Final parameter extraction (whole-clip)
    
    #Jitter/shimmer
    point_process = call(parsel_audio, "To PointProcess (periodic, cc)", minSearchFrequency, maxSearchFrequency)
   
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) #final param!
    shimmer_local = call([parsel_audio, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #final param!
    
    #Speech & articulation rates
    temp_dir = tempfile.mkdtemp()
    try:
        temp_path = os.path.join(temp_dir, os.path.basename(audio_path))
        shutil.copy(audio_path, temp_path)
        scriptRunResult = run_file("scripts/syllable_nuclei.praat", -25, 2, .3, True, temp_dir, capture_output=True)
        scriptOutput = scriptRunResult[1]
        outputLines = [l for l in scriptOutput.strip().split("\n") if l.strip()] #kind of scuffed confusing line but basically creates outputLines as an array of all the nonempty script output lines.
        lineWithData = outputLines[-1]
        outputFields = [f.strip() for f in lineWithData.split(",")] #array of all the data as the script outputs: soundname, nsyll, npause, dur (s), phonationtime (s), speechrate (nsyll/dur), articulation rate (nsyll / phonationtime), ASD (speakingtime/nsyll)
       
        speechRate = float(outputFields[5]) #final param!
        articulationRate = float(outputFields[6]) #final param!
       
    finally:
        shutil.rmtree(temp_dir)

    allExtractedParameters = {**segmentedParameters,
                              "jitter_local":jitter_local, "shimmer_local":shimmer_local,
                              "speechRate":speechRate, "articulationRate":articulationRate
                              }
    cacheKey = (audio_path, numSegments)
    allClipValues[cacheKey] = allExtractedParameters.copy()
    allExtractedParameters = applyParameterExclusions(allExtractedParameters)
    global numParams
    numParams = len(dictValsToArray(allExtractedParameters))
    return allExtractedParameters

def applyParameterExclusions(parameters):
    for p in parametersToExclude:
        parameters.pop(p, None)
        for segmentIndex in range(1, numSegments + 1):
            parameters.pop(f"segment{segmentIndex}_{p}", None)
    return parameters

def dictValsToArray(propertyDict):
    if isinstance(propertyDict, dict):
        values = []
        for value in propertyDict.values():
            values.extend(np.ravel(value))
        return values
    elif isinstance(propertyDict, (list, tuple, np.ndarray)):
        return np.ravel(propertyDict)
    else:
        raise TypeError("the inputted value was neither a list nor a dict.")


allActorNormalizationValues = {}
allGenderActorStds = {0: [], 1: []} # 0 = female, 1 = male
allActorIndivTotalNormVals = {}
def findActorNormalizationValues(actorNumber, audio_path, pathToAllAudios):
    if stdCalculationStyle == 'individual':
        if(actorNumber in allActorNormalizationValues.keys()):
            return allActorNormalizationValues[actorNumber]
        else:
            clipValues = extractNeutralClipParams(actorNumber)
            if clipValues != "clip skipped.":          
                meanValues = [np.mean([clipValues[0][key], clipValues[1][key], clipValues[2][key], clipValues[3][key]], axis=0) for key in clipValues[0].keys()]
                stdValues = [np.std([clipValues[0][key], clipValues[1][key], clipValues[2][key], clipValues[3][key]], axis=0, ddof=1) for key in clipValues[0].keys()]
                actorNormalizationValues = [meanValues, stdValues, clipValues[0], clipValues[1], clipValues[2], clipValues[3]]
                allActorNormalizationValues[actorNumber] = actorNormalizationValues
                return actorNormalizationValues
            else: return "clip skipped."
    elif stdCalculationStyle == 'gender':
        if allGenderActorStds[actorNumber % 2] != []:
            if actorNumber not in allActorNormalizationValues.keys():
                neutralClipParams = extractNeutralClipParams(actorNumber)
                actorMeans = {key: np.mean([clip[key] for clip in neutralClipParams], axis=0) for key in neutralClipParams[0].keys()}
                actorStds = {key: np.std([clip[key] for clip in neutralClipParams], axis=0, ddof=1) for key in neutralClipParams[0].keys()}
                clip1Values, clip2Values, clip3Values, clip4Values = [neutralClipParams[i] for i in range(4)]
                allActorNormalizationValues[actorNumber] = [actorMeans, actorStds, clip1Values, clip2Values, clip3Values, clip4Values]
            return [allActorNormalizationValues[actorNumber][0], allGenderActorStds[actorNumber%2], allActorNormalizationValues[actorNumber][2], allActorNormalizationValues[actorNumber][3], allActorNormalizationValues[actorNumber][4], allActorNormalizationValues[actorNumber][5]]
        else:
            i = actorNumber % 2
            if i == 0: i = 2
            while i <= 24:
                neutralClipParams = extractNeutralClipParams(i)
                if neutralClipParams != "clip skipped.":
                    allGenderActorStds[actorNumber%2].append({key: np.std([clip[key] for clip in neutralClipParams], axis=0, ddof=1) for key in neutralClipParams[0].keys()})
                    if i == actorNumber:
                        actorMeans = {key: np.mean([clip[key] for clip in neutralClipParams], axis=0) for key in neutralClipParams[0].keys()}
                        actorStds = {key: np.std([clip[key] for clip in neutralClipParams], axis=0, ddof=1) for key in neutralClipParams[0].keys()}
                        clip1Values, clip2Values, clip3Values, clip4Values = [neutralClipParams[i] for i in range(4)]
                        allActorNormalizationValues[actorNumber] = [actorMeans, actorStds, clip1Values, clip2Values, clip3Values, clip4Values]
                else: continue
                i += 2
            genderMeanStds = [np.mean([allGenderActorStds[actorNumber%2][j][key] for j in range(len(allGenderActorStds[actorNumber%2]))], axis=0) for key in neutralClipParams[0].keys()]
            return [actorMeans, genderMeanStds, clip1Values, clip2Values, clip3Values, clip4Values]
    elif stdCalculationStyle == 'individual_total':
        if(actorNumber in allActorIndivTotalNormVals):
            return allActorIndivTotalNormVals[actorNumber]
        else:
            allAudioParameters = []
            for c in range(1, 61):
                audio_path = f"{pathToAllAudios}Actor_{actorNumber:02d}\\03-01-{(1 if c <= 4 else (c+3)//8 + 1):02d}-{(1 if c <= 4 else (c-5)//4 % 2 + 1):02d}-{((c-1)//2 % 2 + 1):02d}-{((c-1) % 2 + 1):02d}-{actorNumber:02d}.wav"
                audioInfo = extractInfoFromPath(audio_path)
                audioParameters = extractParameters(audio_path)
                if audioParameters != "clip skipped.":
                    allAudioParameters.append(audioParameters)
                else:
                    continue
            actorMeans = {key: np.mean([clip[key] for clip in allAudioParameters], axis=0) for key in allAudioParameters[0].keys()}
            actorStds = {key: np.std([clip[key] for clip in allAudioParameters], axis=0, ddof=1) for key in allAudioParameters[0].keys()}
            allActorIndivTotalNormVals[actorNumber] = [actorMeans, actorStds] + allAudioParameters
            return allActorIndivTotalNormVals[actorNumber]
    else: raise RuntimeError("The value of stdCalculationStyle was not recognized as an implemented style.")
   
def extractNeutralClipParams(actorNumber):
    formattedActorNumber = f"{actorNumber:02d}"
    clip1Values = extractParameters(audio_path = pathToAllAudios+'Actor_'+formattedActorNumber+'\\03-01-01-01-01-01-'+formattedActorNumber+'.wav')
    clip2Values = extractParameters(audio_path = pathToAllAudios+'Actor_'+formattedActorNumber+'\\03-01-01-01-01-02-'+formattedActorNumber+'.wav')
    clip3Values = extractParameters(audio_path = pathToAllAudios+'Actor_'+formattedActorNumber+'\\03-01-01-01-02-01-'+formattedActorNumber+'.wav')
    clip4Values = extractParameters(audio_path = pathToAllAudios+'Actor_'+formattedActorNumber+'\\03-01-01-01-02-02-'+formattedActorNumber+'.wav')
    if clip1Values == "clip skipped." or clip2Values == "clip skipped." or clip3Values == "clip skipped." or clip4Values == "clip skipped.":
        return "clip skipped."
    else: return [clip1Values, clip2Values, clip3Values, clip4Values]
   
def normalizeParameters(audio_path, actorNumber, audio_parameters):
    normalizationValues = findActorNormalizationValues(actorNumber, audio_path, pathToAllAudios)
    if normalizationValues == "clip skipped." or audio_parameters == "clip skipped.":
            return "clip skipped."
    audio_info = extractInfoFromPath(audio_path)
    if stdCalculationStyle == 'individual':
        if audio_info["emotion"] != "neutral":
            means = normalizationValues[0]
            stds = normalizationValues[1]
        else:
            currentClipIndex = (audio_info["clipNumber"] if audio_info["clipNumber"] == "01" else 4 + 8 * (int(audio_info["emotionNumber"]) - 2) + audio_info["clipNumber"])
            otherNeutralClipValues = normalizationValues[2:].copy()
            otherNeutralClipValues.pop(currentClipIndex)
            means = [np.mean([otherNeutralClipValues[0][key], otherNeutralClipValues[1][key], otherNeutralClipValues[2][key]], axis=0) for key in normalizationValues[2].keys()]
            stds = [np.std([otherNeutralClipValues[0][key], otherNeutralClipValues[1][key], otherNeutralClipValues[2][key]], axis=0, ddof=1) for key in normalizationValues[2].keys()]
   
    elif stdCalculationStyle == 'gender':
        if audio_info["emotion"] != "neutral":
            means = [np.mean(normalizationValues[0][key]) for key in normalizationValues[2].keys()]
            otherActorStds = [v for i, v in enumerate(allGenderActorStds[actorNumber%2]) if (actorNumber // 2 + actorNumber%2 - 1)!=i]
            if currentActorStdWeight == -1:
                stds = [np.mean([allActorNormalizationValues[actorNumber][1][key]] + [otherActorStds[i][key] for i in range(len(otherActorStds))], axis=0) for key in normalizationValues[2].keys()]
            else:
                stds = [currentActorStdWeight * allActorNormalizationValues[actorNumber][1][key] + (1 - currentActorStdWeight) * np.mean([otherActorStds[i][key] for i in range(len(otherActorStds))], axis=0) for key in normalizationValues[2].keys()]
        else:
            otherNeutralClipValues = [v for i, v in enumerate(normalizationValues[2:]) if audio_info["clipNumber"]!=i]
            means = [np.mean([otherNeutralClipValues[0][key], otherNeutralClipValues[1][key], otherNeutralClipValues[2][key]], axis=0) for key in normalizationValues[2].keys()]
            currentActorStds = {key: np.std([otherNeutralClipValues[0][key], otherNeutralClipValues[1][key], otherNeutralClipValues[2][key]], axis=0, ddof=1) for key in normalizationValues[2].keys()}
            otherActorStds = [v for i, v in enumerate(allGenderActorStds[actorNumber%2]) if (actorNumber // 2 + actorNumber%2 - 1)!=i]
            if currentActorStdWeight == -1:
                stds = [np.mean([currentActorStds[key]] + [otherActorStds[i][key] for i in range(len(otherActorStds))], axis=0) for key in normalizationValues[2].keys()]
            else:
                stds = [currentActorStdWeight * currentActorStds[key] + (1 - currentActorStdWeight) * np.mean([otherActorStds[i][key] for i in range(len(otherActorStds))], axis=0) for key in normalizationValues[2].keys()]
    elif stdCalculationStyle == 'individual_total':
        otherClipValues = [v for i, v in enumerate(normalizationValues[2:]) if (audio_info["clipNumber"] if audio_info["emotionNumber"] == "01" else 8 * int(audio_info["emotionNumber"]) + audio_info["clipNumber"] - 12) != i]
        means = [np.mean([otherClipValues[i][key] for i in range(len(otherClipValues))], axis=0) for key in normalizationValues[2].keys()]
        neutralClipStds = {key: np.std([otherClipValues[i][key] for i in range(3 if audio_info["emotionNumber"] == "01" else 4)], axis=0, ddof=1) for key in normalizationValues[2].keys()}
        nonNeutralClipStds = {key: np.std([otherClipValues[i][key] for i in range(3 if audio_info["emotionNumber"] == "01" else 4, len(otherClipValues))], axis=0, ddof=1) for key in normalizationValues[2].keys()}
        if actorNeutralClipStdWeight == -1:
            stds = [np.mean([neutralClipStds[key]] + [nonNeutralClipStds[key]], axis=0) for key in normalizationValues[2].keys()]
        else:
            stds = [actorNeutralClipStdWeight * np.mean([neutralClipStds[key]], axis=0) + (1 - actorNeutralClipStdWeight) * np.mean([nonNeutralClipStds[key]], axis=0) for key in normalizationValues[2].keys()]
    else: raise RuntimeError("The value of stdCalculationStyle was not recognized as an implemented style.")
   
    zScoredValues = {}
    for i, key in enumerate(normalizationValues[2].keys()):
        zScoredValues[key] = (audio_parameters[key] - means[i])/stds[i]
    return zScoredValues
   
def setParametersToExclude(newParamsToExclude):
    global parametersToExclude
    global numParamsChanged
    if parametersToExclude != newParamsToExclude:    
        parametersToExclude = newParamsToExclude
        numParamsChanged = True
def setNumMFCCs(newNumMFCCs):
    global numMFCCs
    global numParamsChanged
    if numMFCCs != newNumMFCCs:    
        numMFCCs = newNumMFCCs
        numParamsChanged = True
def setStdCalculationStyle(newStdCalculationStyle):
    global stdCalculationStyle
    global numParamsChanged
    if stdCalculationStyle != newStdCalculationStyle:    
        stdCalculationStyle = newStdCalculationStyle
        numParamsChanged = True
def setCurrentActorStdWeight(newActorStdWeight):
    global currentActorStdWeight
    global numParamsChanged
    if currentActorStdWeight != newActorStdWeight:    
        currentActorStdWeight = newActorStdWeight
        numParamsChanged = True
def setActorNeutralClipStdWeight(newActorNeutralClipStdWeight):
    global actorNeutralClipStdWeight
    global numParamsChanged
    if actorNeutralClipStdWeight != newActorNeutralClipStdWeight:    
        actorNeutralClipStdWeight = newActorNeutralClipStdWeight
        numParamsChanged = True
def setNumSegments(newNumSegments):
    global numSegments
    global numParamsChanged
    if numSegments != newNumSegments:    
        numSegments = newNumSegments
        numParamsChanged = True