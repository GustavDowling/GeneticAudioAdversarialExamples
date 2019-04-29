
import csv
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

def writetocsv(filepath,data):
    with open(filepath,'w') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerows(data)
def Wrongindex(inputa,best_text,target_phrase):
        l=len(inputa)
        ind1=0
        ind2=len(target_phrase)
        targ_len=len(target_phrase)
        for i in range(targ_len):
            if best_text[i]==target_phrase[i]:
                ind1+=1
            else:
                break
            if i>=len(best_text)-1:
                break
        for i in range(-1,ind1-targ_len,-1):
            if best_text[i]==target_phrase[i]:
                ind2-=1
            else:
                break
            if -i>=len(best_text):
                break
        startind=round(l*(ind1/targ_len))
        endind=round(l*(1-ind2/targ_len))
        return [startind,endind]

def MfccDist(audio1,audio2,fs,numcep):
    coe1=mfcc(audio1,samplerate=fs,numcep=numcep)
    coe2=mfcc(audio2,samplerate=fs,numcep=numcep)

    diff=0
    for i in range(len(coe1)):
        diff+=sum(np.square(coe1[i]-coe2[i]))
    return diff



def Similarity(audio1,audio2):
    corr = np.corrcoef([audio1, audio2])[0][1]

    return corr


def largestdev(audio1,audio2):
    diff=np.subtract(audio1,audio2)
    diff=np.abs(diff)
    return np.max(diff)
