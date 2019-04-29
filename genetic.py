import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import os
import sys
sys.path.append("DeepSpeech")
import random
from scipy.signal import butter, lfilter
#from matplotlib import pyplot as plt
import helpfuncs


###########################################################################
# This section of code is credited to:
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.

#LICENSE

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met: 

#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer. 
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution. 

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.

tf.load_op_library = lambda x: x
generation_tmp = os.path.exists
os.path.exists = lambda x: True

class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]

class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v

tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = generation_tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None

from util.text import ctc_label_dense_to_sparse
from tf_logits import compute_mfcc, get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"
#######################################################################

#######################################################################
# This section of code is credited to:
## Copyright (c) 2018 Rohan Taori

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



def db(audio):
    if len(audio.shape) > 1:
        maxx = np.max(np.abs(audio), axis=1)
        return 20 * np.log10(maxx) if np.any(maxx != 0) else np.array([0])
    maxx = np.max(np.abs(audio))
    return 20 * np.log10(maxx) if maxx != 0 else np.array([0])

def load_wav(input_wav_file):
    # Load the inputs that we're given
    fs, audio = wav.read(input_wav_file)
    assert fs == 16000
    print('source dB', db(audio))
    return audio

def save_wav(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16))
    print('output dB', db(audio))


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# also the highpass_filter method below. 
##################################################################################################
    
def cut_low_noise(array,x=10):
    # takes an input array and sets all values below x% of the max  
    cutoff = np.amax(array)*x/100  # x%
    new_array = np.copy(array)
    for i in range(len(array)):
        if array[i] <= cutoff and array[i] >= -cutoff:
            new_array[i] = 0
        else:
            new_array[i] = array[i]
    return new_array
                  

def target_highs(array,x=10):
    
    # takes an input array and sets all values below x% of the max to 0 and the others to 1  
    cutoff = np.amax(array)*x/100  # x%
    new_array = np.copy(array)
    for i in range(len(array)):
        if array[i] <= cutoff and array[i] >= -cutoff:
            new_array[i] = 0
        else:
            new_array[i] = 1
    return new_array


class GeneticAttack():

    def __init__(self,inpath,outpath,target_phrase):
        #Here defines the basic parameters of the genetic attack
        self.inpath=inpath
        self.outpath=outpath
        self.target_phrase=target_phrase

        self.popsize=100
        self.elite_size=10
        self.mut_prob=0.005
        self.noisestd=40

        self.max_iters = 5000
        self.num_points_estimate = 100

        self.input_audio = load_wav(inpath).astype(np.float32)
        self.pop = np.expand_dims(self.input_audio, axis=0)
        self.pop = np.tile(self.pop, (self.popsize, 1))
        
        self.highs = target_highs(self.input_audio,50)
        print("SELF HIGHS TEST: ",np.sum(self.highs))

        self.funcs = self.setup_graph(self.pop, np.array([toks.index(x) for x in target_phrase]))

    def highpass_filter(self,data, cutoff=7000, fs=16000, order=10):

        b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
        return lfilter(b, a, data)


    def initpop(self):
        popshape=self.pop.shape
        noise=np.random.randn(*popshape)*self.noisestd*0.05
        noise=self.highpass_filter(noise)
        noise=cut_low_noise(noise,10)
        for i in range(len(noise)):
            noise[i] = self.highs[i]*noise[i]
            
        self.pop=np.add(self.pop,noise)

    def crossover(self,n1,n2):
        for i in range(len(n1)):
            if np.random.random()<0.5:
                n1[i]=n2[i]
        return n1

    def Mutation(self,data,eps_max,index=None):
        noise=np.random.randn(*data.shape)*eps_max
        noise=self.highpass_filter(noise)
        #print('noise=',noise)
        if index==None:
            for i in range(len(data)):
                if np.random.random()<self.mut_prob:
                    data[i]=data[i]+noise[i]
        else:
            for i in range(index[0],index[1]):
                if np.random.random()<self.mut_prob*2:
                    data[i]=data[i]+noise[i]
            for i in range(index[0]):
                if np.random.random()<(self.mut_prob):
                    data[i]=data[i]+noise[i]
            for i in range(index[1],len(data)):
                if np.random.random()<(self.mut_prob):
                    data[i]=data[i]+noise[i]


        return data
    def Wrongindex(self,best_text):
        l=len(self.input_audio)
        ind1=0
        ind2=len(self.target_phrase)
        targ_len=len(self.target_phrase)
        for i in range(targ_len):
            if best_text[i]==self.target_phrase[i]:
                ind1+=1
            else:
                break
            if i>=len(best_text)-1:
                break
        for i in range(-1,ind1-targ_len,-1):
            if best_text[i]==self.target_phrase[i]:
                ind2-=1
            else:
                break
            if -i>=len(best_text):
                break
        startind=round(l*(ind1/targ_len))
        endind=round(l*(1-ind2/targ_len))
        return [startind,endind]




    def update_prob(self,old_score,score):
        alpha=0.9
        beta=0.001
        currScore=max(score)
        #print('currscore',currScore)


        prevScore=max(old_score)
        #print('prevScore',prevScore)
        diff=currScore-prevScore
        diff=abs(diff/currScore)

        if diff!=0:
            prob=(alpha+beta/diff)*self.mut_prob
            if prob>0.02:
                self.mut_prob=0.02
            elif prob<0.002:
                self.mut_prob=0.002
            else:
                self.mut_prob=prob

            nstd=(alpha+beta/diff)*self.noisestd
            if nstd>60:
                self.noisestd=60
            elif nstd<20:
                self.noisestd=20
            else:
                self.noisestd=nstd
        # if diff!=0:
        #     self.mut_prob=max(0.001,alpha*self.mut_prob+(beta/diff))









    def setup_graph(self, input_audio_batch, target_phrase):
            batch_size = input_audio_batch.shape[0]
            weird = (input_audio_batch.shape[1] - 1) // 320
            logits_arg2 = np.tile(weird, batch_size)
            dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
            dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)

            pass_in = np.clip(input_audio_batch, -2 ** 15, 2 ** 15 - 1)
            seq_len = np.tile(weird, batch_size).astype(np.int32)

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                inputs = tf.placeholder(tf.float32, shape=pass_in.shape, name='a')
                len_batch = tf.placeholder(tf.float32, name='b')
                arg2_logits = tf.placeholder(tf.int32, shape=logits_arg2.shape, name='c')
                arg1_dense = tf.placeholder(tf.float32, shape=dense_arg1.shape, name='d')
                arg2_dense = tf.placeholder(tf.int32, shape=dense_arg2.shape, name='e')
                len_seq = tf.placeholder(tf.int32, shape=seq_len.shape, name='f')

                logits = get_logits(inputs, arg2_logits)
                target = ctc_label_dense_to_sparse(arg1_dense, arg2_dense, len_batch)
                ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=len_seq)
                decoded, _ = tf.nn.ctc_greedy_decoder(logits, arg2_logits, merge_repeated=True)

                sess = tf.Session()
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, "models/session_dump")

            func1 = lambda a, b, c, d, e, f: sess.run(ctcloss,
                                                      feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d,
                                                                 arg2_dense: e, len_seq: f})
            func2 = lambda a, b, c, d, e, f: sess.run([ctcloss, decoded],
                                                      feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d,
                                                                 arg2_dense: e, len_seq: f})
            return (func1, func2)

    def getctcloss(self, input_audio_batch, target_phrase, decode=False):
            batch_size = input_audio_batch.shape[0]
            weird = (input_audio_batch.shape[1] - 1) // 320
            logits_arg2 = np.tile(weird, batch_size)
            dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
            dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)

            pass_in = np.clip(input_audio_batch, -2 ** 15, 2 ** 15 - 1)
            seq_len = np.tile(weird, batch_size).astype(np.int32)

            if decode:
                return self.funcs[1](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
            else:
                return self.funcs[0](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)

    def getFitnessScores(self,input_audio_batch,target_phrase,classify=False):
        #The first score is the ctcloss
        target_enc = np.array([toks.index(x) for x in target_phrase])
        if classify:
            ctcloss, decoded = self.getctcloss(input_audio_batch, target_enc, decode=True)
            all_text = "".join([toks[x] for x in decoded[0].values])
            index = len(all_text) // input_audio_batch.shape[0]
            final_text = all_text[:index]
        else:
            ctcloss = self.getctcloss(input_audio_batch, target_enc)
        score = -ctcloss
        MfccScore=[]
        Similarity=[]
        LargestDev=[]
        for i in range(len(input_audio_batch)):
            mfccScore=helpfuncs.MfccDist(input_audio_batch[i],self.input_audio,16000,9)
            #print(mfccScore)
            mfccScore=np.multiply(-1,mfccScore)
            MfccScore.append(mfccScore)
            Similarity.append(helpfuncs.Similarity(input_audio_batch[i],self.input_audio))

            largestDev=np.multiply(-1,helpfuncs.largestdev(input_audio_batch[i],self.input_audio))
            LargestDev.append(largestDev)

        #Apparently we need to normalize the score functions

        minimum = np.min(MfccScore)
        maximum = np.max(MfccScore)
        d = maximum - minimum
        if d!=0:
            for i in range(len(MfccScore)):
                MfccScore[i]=(MfccScore[i]-minimum)/d
        else:
            for i in range(len(MfccScore)):
                MfccScore[i]=1

        minimum = np.min(Similarity)
        maximum = np.max(Similarity)
        d = maximum - minimum
        if d!=0:
            for i in range(len(Similarity)):
                Similarity[i] = (Similarity[i] - minimum) / d
        else:
            for i in range(len(Similarity)):
                Similarity[i] = 1

        minimum = np.min(LargestDev)
        maximum = np.max(LargestDev)
        d = maximum - minimum
        if d!=0:
            for i in range(len(LargestDev)):
             LargestDev[i] = (LargestDev[i] - minimum) / d
        else:
            for i in range(len(LargestDev)):
                LargestDev[i]=1
        if classify:
            return (score, MfccScore,Similarity,LargestDev,final_text)
        return score, -ctcloss,MfccScore,Similarity,LargestDev

    def SelectProbability(self,scores):
        #This method returns a list of probability distribution
        minimum=np.min(scores)
        p=[]
        s=0
        for i in range(len(scores)):
            s+=scores[i]-minimum
        if s==0:
            p=[1/len(scores) for i in range(len(scores))]
            return p
        for i in range(len(scores)):
            temp=scores[i]-minimum
            p.append(temp/s)
        return p

    def RWGA(self,score1,score2,score3):
        n1=np.random.random()
        n2=np.random.random()
        n3=np.random.random()
        s=n1+n2+n3
        w1=n1/s
        w2=n2/s
        w3=n3/s
        score=[]
        for i in range(len(score1)):
            temp=w1*score1[i]+w2*score2[i]+w3*score3[i]
            score.append(temp)
        return score



    def selectParents(self,score,MfccScore,Simi,LargestDev=None):
        # temp_score=np.copy(score)
        #
        #
        # #print(temp_score)
        # parents_temp=np.empty((self.elite_size*2,self.pop.shape[1]))
        # SimiElite=[]
        # parents= np.empty((self.elite_size, self.pop.shape[1]))
        # #Here we add the gene with highest score
        # max_fit_index = np.where(temp_score == np.max(temp_score))
        # max_fit_index = max_fit_index[0][0]
        # parents[0]=self.pop[max_fit_index]
        # temp_score[max_fit_index]=-99999
        # for i in range(round(self.elite_size*1.5)):
        #     max_fit_index=np.where(temp_score==np.max(temp_score))
        #     #print('maxindex=',max_fit_index)
        #     max_fit_index=max_fit_index[0][0]
        #     parents_temp[i]=self.pop[max_fit_index]
        #     temp_score[max_fit_index]=-99999
        #     SimiElite.append(Simi[max_fit_index])
        # indx=np.argsort(SimiElite)[-self.elite_size+1:]
        #
        # parents[1:]=parents_temp[indx]


        #Here is the multiple objective version
        MfccScore=np.array(MfccScore)
        Simi=np.array(Simi)
        LargestDev=np.array(LargestDev)
        temp_score=np.copy(score)
        parents_temp=np.empty((self.elite_size*2-1,self.pop.shape[1]))
        index_=[x for x in range(self.elite_size*2-1)]
        parents = np.empty((self.elite_size, self.pop.shape[1]))
        idx=np.argsort(temp_score)[-self.elite_size*2:]
        #print('idx',idx)
        parents[0]=self.pop[idx[-1]] #Add the one with the highest CTC loss to next gen
        parents_temp=self.pop[idx[:-1]]
        #print('mfcc',MfccScore[2,3,4])
        #print('0',idx[:-1])
        #print('1',MfccScore[idx[:-1]])
        Score=self.RWGA(MfccScore[idx[:-1]],Simi[idx[:-1]],LargestDev[idx[:-1]])
        print('Score:',Score)
        prop=self.SelectProbability(Score)
        index=np.random.choice(index_,self.elite_size-1,replace=False,p=prop)
        parents[1:]=parents_temp[index]





        return parents
    def GenerateOffSprings(self,parents,wrongindx=None):
        offspring=np.empty((self.popsize-parents.shape[0],parents.shape[1]))
        for i in range(offspring.shape[0]):
            parents1_indx=i%parents.shape[0]
            parents2_indx=(i+1)%parents.shape[0]
            temp=self.crossover(parents[parents1_indx],parents[parents2_indx])
            temp=self.Mutation(temp,self.noisestd,wrongindx)
            offspring[i]=temp
        return offspring

    def attack(self):
        max_fitness_score = float('-inf')

        iter=1

        log=[]
        best_text=''
        prev_score=[0]

        log.append(['Generations','Best CTC','Decode as','Similarity','Edit Distance','Lowest MFccDist'])
        mut_index=None
        while iter<self.max_iters and best_text!=self.target_phrase:


            pop_scores, ctc,MfccScore,Simi,LargestDev= self.getFitnessScores(self.pop, self.target_phrase)
            best_index = np.argsort(pop_scores)[-1]
            best_ctc = ctc[best_index]
            lowest_mfcc=min(MfccScore)

            parents=self.selectParents(pop_scores,MfccScore,Simi,LargestDev)
            offsprings=self.GenerateOffSprings(parents,mut_index)
            self.pop[0:parents.shape[0]]=parents
            #print('parents=',parents)
            self.pop[parents.shape[0]:]=offsprings
            #print('offsprings',offsprings)
            if iter!=1:
                self.update_prob(prev_score,pop_scores)
            prev_score=pop_scores
            print(self.noisestd,self.mut_prob)

            if iter%10==0:
                print('**************************** ITERATION {} ****************************'.format(iter))
                print("current loss is:{}".format(-best_ctc))


                bestpop = np.tile(np.expand_dims(self.pop[best_index], axis=0), (self.popsize, 1))
                _,__,___,____,best_text=self.getFitnessScores(bestpop,self.target_phrase,classify=True)
                corr = "{0:.4f}".format(np.corrcoef([self.input_audio, self.pop[best_index]])[0][1])
                dist = levenshteinDistance(best_text, self.target_phrase)
                print('Currently decoded as: {}'.format(best_text))
                print("similarity:",corr)
                print('Dist=',dist)
                print('Lowest MFCC Dist is', lowest_mfcc)
                save_wav(bestpop[0], self.outpath)
                log.append([iter,-best_ctc,best_text,corr,dist,lowest_mfcc])
                if dist<4:
                    mut_index=self.Wrongindex(best_text)
                else:
                    mut_index=None

                if best_text == self.target_phrase:
                    bestpop = np.tile(np.expand_dims(self.pop[best_index], axis=0), (self.popsize, 1))
                    _, __,__,____,best_text = self.getFitnessScores(bestpop, self.target_phrase, classify=True)
                    print(best_text)
                    if best_text==self.target_phrase:
                        save_wav(bestpop[0], 'final.wav')

                        print("finished, saving file as: ",self.outpath)


                        return log
            iter+=1


        return log

inpath='sample11.wav'
outpath='sample11_adv.wav'
target_phrase='will this be noisy'
print('target phrase:', target_phrase)
print('source file:', inpath)
g=GeneticAttack(inpath,outpath,target_phrase)
log=g.attack()
helpfuncs.writetocsv('AttackLog_11.csv',log)



















