# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 08:23:43 2024

@author: yhzha
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.losses import mse
from tensorflow.keras.losses import binary_crossentropy
import warnings
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler

warnings.filterwarnings('ignore') 

import os
from scipy.stats import zscore

def remove_outliers(df, column_name):
    df['z_score'] = zscore(df[column_name])
    df = df[df['z_score'].abs() <= 3]
    df = df.drop(columns=['z_score'])
    return df

def dataProcess(dataFile, treatmentV, confounders, outcomes, outType):
    dataNhanes = pd.read_csv(dataFile)
    
    dataNhanes = dataNhanes.loc[dataNhanes['RIDAGEYR'] >= 20,:]

    useCol = [treatmentV] + confounders + [outcomes]
    dataNhanes = dataNhanes.loc[:,useCol]
    
    dataNhanes['z_score'] = zscore(dataNhanes[outcomes])
    dataNhanes = dataNhanes[dataNhanes['z_score'].abs() <= 3]
    
    dataNhanes['z_score'] = zscore(dataNhanes[treatmentV])
    dataNhanes = dataNhanes[dataNhanes['z_score'].abs() <= 3]
    
    dataNhanes = dataNhanes.drop(columns=['z_score'])
    
    # 'discrete' 'continuous'
    useColType = {}
    for col_ in useCol:
        useColType[col_] = 'continuous'
    
    useColType[outcomes] = outType
    
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    
    continuousV = [key for key, value in useColType.items() if value == 'continuous']
    stasDict = {'min': dataNhanes[continuousV].min(),
                'max': dataNhanes[continuousV].max(),
                'mean': dataNhanes[continuousV].mean(),
                'std': dataNhanes[continuousV].std()}
    
    dataNhanes.loc[:,continuousV] = scaler.fit_transform(dataNhanes.loc[:,continuousV])

    if useColType[outcomes] == 'discrete':
        yEncode = LabelEncoder().fit_transform(dataNhanes.loc[:,outcomes])
    
    
    inputT = dataNhanes.loc[:,treatmentV].reset_index(drop=True)
    inputC = dataNhanes.loc[:,confounders].reset_index(drop=True)
    inputY = dataNhanes.loc[:,outcomes].reset_index(drop=True)
    
    plt.scatter(dataNhanes.loc[0:50,treatmentV], dataNhanes.loc[0:50,outcomes])
    plt.xlabel('treatment')
    plt.ylabel('outcomes')
    plt.show() 

    return dataNhanes, stasDict, inputT, inputC, inputY

# Size of noise vector
def get_discriminator_model():
    t_input = layers.Input(shape = (tDim,))
    x_random_input = layers.Input(shape = (rep_dim,))
    
    is_indep_x = layers.Concatenate()([t_input,x_random_input])
    
    is_true_ = layers.Dense(intermediate_dim,activation = 'elu')(is_indep_x)
    is_true_ = layers.Dense(intermediate_dim,activation = 'elu')(is_true_)
    
    is_true = layers.Dense(1)(is_true_)
    
    d_model = keras.models.Model([t_input,x_random_input], is_true, name="discriminator")
    return d_model


"""
## Create the generator
"""
def get_generator_model():
    c = layers.Input(shape=(cDim,))
 
    x_rep_ = layers.Dense(intermediate_dim * 2, activation='elu')(c)
    x_rep_ = layers.Dense(intermediate_dim, activation='elu')(x_rep_)
    x_rep_ = layers.Dense(intermediate_dim, activation='elu')(x_rep_)
    x_rep = layers.Dense(rep_dim,activation='elu', name = 'x_rep')(x_rep_)
    
    g_model = keras.models.Model(c, x_rep, name="generator")
    return g_model

def generalizationSCM():
    t = layers.Input(shape=(tDim,))
    x_rep = layers.Input(shape=(rep_dim,))
    
    x_rep_int_ = layers.Dense(round(1.5 * rep_dim) , activation='elu',kernel_regularizer = 'l1')(x_rep)
    x_rep_int_ = layers.Dense(round(rep_dim) , activation='elu',kernel_regularizer = 'l1')(x_rep_int_)
    x_rep_int = layers.Dense(round(0.5 * rep_dim) , activation='elu',kernel_regularizer = 'l1')(x_rep_int_)
    
    t_int_ = layers.Dense(3 , activation='elu')(t)
    t_int = layers.Dense(2 , activation='elu')(t_int_)
    
    y_x = layers.Concatenate()([x_rep_int,t_int])
    
    y_ = layers.Dense(3 , activation='elu',kernel_regularizer = 'l1')(y_x)
    y_ = layers.Dense(1 , activation='elu',kernel_regularizer = 'l1')(y_)
    
    y = layers.Dense(1)(y_)
    
    # model
    GSCM = keras.models.Model([t,x_rep], y, name="generator")
    return GSCM

## Create a WGAN-GP model

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        GSCM,
        rep_dim,
        discriminator_extra_steps = 10,
        generator_extra_steps = 3,
        GSCM_extra_steps = 5,
        gp_weight = 0.5,
        GSCM_weight = 1e-5 
        
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.GSCM = GSCM
        self.rep_dim = rep_dim
        self.d_steps = discriminator_extra_steps
        self.g_steps = generator_extra_steps
        self.GSCM_steps = GSCM_extra_steps
        self.gp_weight = gp_weight,
        self.GSCM_weight = GSCM_weight
        

    def call(self, real_data, training=False):
        real_t_ = real_data[0]
        c_ = real_data[1]
        
        #with tf.GradientTape() as tape:
        # Generate fake images from the latent vector
        generate_x = self.generator(c_)
        generate_y = self.GSCM([real_t_,generate_x])
        # Get the logits for the fake images
        # fake_logits = self.discriminator(fake_t_y, training=training)
        # Get the logits for real images
        # real_logits = self.discriminator([real_t,real_y], training=training)
        return generate_x,generate_y

    def compile(self, d_optimizer, g_optimizer, GSCM_optimizer, d_loss_fn, g_loss_fn,factual_loss):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.GSCM_optimizer = GSCM_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.factual_loss = factual_loss
        
    def gradient_penalty(self, batch_size, x_random_rep, gene_x_rep, real_t):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff_t = gene_x_rep - x_random_rep
        interpolated = x_random_rep + alpha * diff_t
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([real_t,interpolated], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads)))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    def train_step(self, real_data):
        # Get the batch size

        real_t = real_data[0][0]
        c = real_data[0][1]
        y = real_data[0][2]
        
        # real batch size
        batch_size = tf.shape(real_t)[0]
        
        for i in range(self.d_steps):
            # Get the latent vector

            x_random_rep = tf.random.normal(
                shape=(batch_size, self.rep_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                gene_x_rep = self.generator(c)
                
                # Get the logits for the fake images
                fake_logits = self.discriminator([real_t,gene_x_rep])
                # Get the logits for real images
                real_logits = self.discriminator([real_t,x_random_rep])

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_logits, fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, x_random_rep, gene_x_rep,real_t)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight


            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator now.
        # Get the latent vector
        for i in range(self.g_steps):
            with tf.GradientTape() as tape:
                # Generate fake images using the generator
                # gene_x_rep = self.generator(c, training=True)
                gene_x_rep = self.generator(c)
                
                # Get the discriminator logits for fake images
                gen_logits = self.discriminator([real_t,gene_x_rep])
                
                # save item
                cf_y_g = self.GSCM([real_t,gene_x_rep])
                
                # Calculate the generator loss
                g_loss_ = self.g_loss_fn(gen_logits)
                
                # save item
                GSCM_loss_G = self.factual_loss(cf_y_g,y)
                g_loss = g_loss_ + GSCM_loss_G * self.GSCM_weight
                
            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(
                zip(gen_gradient, self.generator.trainable_variables)
            )
        
        # Train the generator now.
        # Get the latent vector
        for i in range(self.GSCM_steps):
            with tf.GradientTape() as tape:
                # Generate fake images using the generator
                gene_x_rep = self.generator(c)
                
                # Get couterfactual
                cf_y = self.GSCM([real_t,gene_x_rep])
                # Calculate the generator loss
                GSCM_loss = self.factual_loss(cf_y,y)
                
            # Get the gradients w.r.t the generator loss
            GSCM_gradient = tape.gradient(GSCM_loss, self.GSCM.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.GSCM_optimizer.apply_gradients(
                zip(GSCM_gradient, self.GSCM.trainable_variables)
            )
            
        return {"d_loss": d_loss, "g_loss": g_loss,"GSCM_loss": GSCM_loss}

def discriminator_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return fake_loss - real_loss

# Define the loss functions to be used for generator
def generator_loss(gen_img_logits):
    return -tf.reduce_mean(gen_img_logits)

def factual_loss(fac_y, y):
    return tf.reduce_mean(mse(fac_y,y))

def DRLNets():
    d_model = get_discriminator_model()
    g_model = get_generator_model()
    GSCM = generalizationSCM()
    
    generator_optimizer = keras.optimizers.Adam() 
    discriminator_optimizer = keras.optimizers.Adam()
    GSCM_optimizer = keras.optimizers.Adam(learning_rate = 0.0002)

    # Get the wgan model
    wgan = WGAN(
        discriminator = d_model,
        generator = g_model,
        GSCM = GSCM,
        rep_dim = rep_dim,
    )

    # Compile the wgan model
    wgan.compile(
        d_optimizer = discriminator_optimizer,
        g_optimizer = generator_optimizer,
        GSCM_optimizer = GSCM_optimizer,
        g_loss_fn = generator_loss,
        d_loss_fn = discriminator_loss,
        factual_loss = factual_loss
    )
    
    return wgan

def eveluation(Model, stasD, scalerMod = 'StandardScaler'):
    
    history = Model.fit([inputT, inputC, inputY],batch_size = batch_size, epochs = epochs, verbose = 0)
    
    plt.plot(list(history.history.values())[0])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['d_loss'], loc='upper left')
    plt.show()
    
    plt.plot(list(history.history.values())[1])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['g_loss'], loc='upper left')
    plt.show()
    
    plt.plot(list(history.history.values())[2])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['GSCM_loss'], loc='upper left')
    plt.show()
    
    predict = Model.predict([inputT, inputC])
    
    x_reo_ = predict[0]
    data_later = np.concatenate([inputT.values.reshape(len(inputT),1),x_reo_],axis=1)
    corrcoef_later = np.corrcoef(data_later,rowvar = False)
    sns.heatmap(corrcoef_later,annot=True,linewidths = .5,cmap ="YlGnBu")
    # plt.savefig("hostLater.png",dpi=500,bbox_inches = 'tight')
    plt.show()
    
    print("later cor",np.mean(np.abs(corrcoef_later[1:rep_dim + 1, 0])))
    
    later_columns = ['x_{}'.format(i) for i in range(rep_dim)]
    later_columns.insert(0,'t')
    
    ## train
    GSCM_predict = predict[1]
    plt.plot(inputY[:50])
    plt.plot(GSCM_predict[:50])
    plt.legend(['True_y', 'predict_y'], loc='upper left')
    plt.show()
    
    # inputT, inputC, inputY
    minT = min(inputT)
    maxT = max(inputT)
    print(minT, maxT)
    
    minPerT = np.percentile(inputT, 10)
    maxPerT = np.percentile(inputT, 90)
    print(minPerT, maxPerT)
    
    SpacePointsNum = 20
    treatSpace = np.linspace(minPerT, maxPerT, SpacePointsNum).reshape(SpacePointsNum,-1)
    
    if scalerMod == 'StandardScaler':
        treatSpaceR = [trea_ * stasD['std'][treatmentV] + stasD['mean'][treatmentV] for trea_ in treatSpace]
    else:
        dicT = stasD['max'][treatmentV] - stasD['min'][treatmentV]
        treatSpaceR = [trea_ * dicT + stasD['min'][treatmentV] for trea_ in treatSpace]
        
    MTEF_train = []
    preTargeF_train = []
    for T_ in treatSpace:
        inputT_ = pd.Series(np.repeat(T_,len(inputT))) 
        preTargeF = Model.predict([inputT_, inputC])[1]
        
        factualT_ = pd.Series(np.repeat(T_ + 0.1,len(inputT))) 
        preTargeCF = Model.predict([factualT_, inputC])[1]
    
        if scalerMod == 'StandardScaler':
            preTargeF = [preTargeF_ * stasD['std'][outcomes] + stasD['mean'][outcomes] for preTargeF_ in preTargeF]
            preTargeCF = [preTargeCF_ * stasD['std'][outcomes] + stasD['mean'][outcomes] for preTargeCF_ in preTargeCF]
        else:
            dicO = stasD['max'][outcomes] - stasD['min'][outcomes]
            
            preTargeF = [preTargeF_ * dicO + stasD['min'][outcomes] for preTargeF_ in preTargeF]
            preTargeCF = [treatSpaceR_ * dicO + stasD['min'][outcomes] for treatSpaceR_ in treatSpaceR]
        
        # MTEF = np.mean(predict_y - preTargeCF)
        MTEF = round(np.mean((np.array(preTargeCF) - np.array(preTargeF)) / 0.1), 6)
        MTEF_train.append(MTEF)
        preTargeF_train.append(round(np.mean(np.array(preTargeF)),6))
        
    
    plt.subplots(figsize = (8,6))
    plt.plot(treatSpaceR, preTargeF_train)
    plt.xlabel('treatSpace')
    plt.ylabel('Outcome')
    plt.title('Predict Curve of {}!'.format(treatmentV))
    plt.show()
    
    plt.subplots(figsize = (8,6))
    plt.plot(treatSpaceR, MTEF_train)
    plt.xlabel('treatSpace')
    plt.ylabel('MTEF')
    plt.title('MTEF of {}!'.format(treatmentV))
    plt.show()
    
    return treatSpaceR, preTargeF_train, MTEF_train


"""
1、
"""

dataFile = '../dataset/NhanesDataPro.csv'
dataNHANES = pd.read_csv(dataFile)
dataNHANES.columns

treatmentV = 'LBDSTRSI'
confounders = ['LBXSGTSI','LBXSATSI', 'LBXSASSI','RIDAGEYR']
outcomes = 'LBDGLUSI'
outcomesType = 'continuous'
scalerMod = 'StandardScaler'

Nhanes, stasDict, inputT, inputC, inputY = dataProcess(dataFile, treatmentV, confounders, outcomes, outcomesType)
describe = Nhanes.describe()

corrspear = Nhanes.corr(method='spearman')
plt.subplots(figsize = (10,5))
hosts1=sns.heatmap(corrspear,annot=True,linewidths = .5,cmap ="YlGnBu")
plt.show()
print(np.mean(np.abs(corrspear.loc[treatmentV,confounders])))
    

"""
2、
"""
tDim = 1
cDim = 4
yDim = 1

intermediate_dim = 6
rep_dim = 4
batch_size = 2000
epochs = 100

"""
3、
"""

FacPre_DataFrame = pd.DataFrame()
MTEF_DataFrame = pd.DataFrame()
for i in range(10):
    wgan = DRLNets()
    treatSpace, preTargeF, MTEF = eveluation(Model = wgan, stasD = stasDict, scalerMod = 'StandardScaler')
    
    FacPre_DataFrame['FacPre-{}'.format(i)] = preTargeF
    MTEF_DataFrame['MTEF-{}'.format(i)] = MTEF

saveFacPreFile = '../dataset/1_FacPre_NHANES.csv'
saveMTEFFile = '../dataset/2_MTEF_NHANES.csv'
FacPre_DataFrame.to_csv(saveFacPreFile,index = False)
MTEF_DataFrame.to_csv(saveMTEFFile,index = False)

treatSpace_DataFrame = pd.DataFrame(np.array(treatSpace))
treatSpace_DataFrame.columns = ['treatSpace']
savetreatSpaceFile = '../dataset/0_treatSpace_NHANES.csv'
treatSpace_DataFrame.to_csv(savetreatSpaceFile,index = False)
