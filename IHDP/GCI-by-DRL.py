# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:40:16 2024

@author: yhzha
"""



import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import warnings
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore') 

import os

############################
# Case 
dataFile = './data-ihdp-continuous.csv'
moniData = pd.read_csv(dataFile)
moniColumns = moniData.columns

# Total
# AnsF
x_name = ['ein','pen']
x_dim = 2
a_name = ['X_2','X_11']
a_dim = 2
############################


def splitData(data, x_name, x_dim = 5):
    dataLinearGaussian = data.copy()
    
    t_column = 't'
    c_column = x_name
    a_column = a_name
    y_column = 'y'
    
    y_true_column = 'y_f'
    y_delta_column = 'y_cf'
    
    train_val_rate = 0.8
    train_val_samples = round(len(dataLinearGaussian) * train_val_rate)
    
    random.seed(20180808)
    train_val_sample_select = random.sample(range(len(dataLinearGaussian)), train_val_samples)
    test_sample_select = list(set(range(len(dataLinearGaussian))) - set(train_val_sample_select))
    
    print("----split dataSet----")
    input_t = dataLinearGaussian.loc[train_val_sample_select,t_column].reset_index(drop=True)
    input_c = dataLinearGaussian.loc[train_val_sample_select,c_column].reset_index(drop=True)
    input_a = dataLinearGaussian.loc[train_val_sample_select,a_column].reset_index(drop=True)
    input_y = dataLinearGaussian.loc[train_val_sample_select,y_column].reset_index(drop=True)

    y_true = dataLinearGaussian.loc[train_val_sample_select,y_true_column].reset_index(drop=True)
    y_delta = dataLinearGaussian.loc[train_val_sample_select,y_delta_column].reset_index(drop=True)

    input_t_test = dataLinearGaussian.loc[test_sample_select,t_column].reset_index(drop=True)
    input_c_test = dataLinearGaussian.loc[test_sample_select,c_column].reset_index(drop=True)
    input_a_test = dataLinearGaussian.loc[test_sample_select,a_column].reset_index(drop=True)
    input_y_test = dataLinearGaussian.loc[test_sample_select,y_column].reset_index(drop=True)
    
    y_true_test = dataLinearGaussian.loc[test_sample_select,y_true_column].reset_index(drop=True)
    y_delta_test = dataLinearGaussian.loc[test_sample_select,y_delta_column].reset_index(drop=True)
    
    return input_t,input_c,input_a,input_y,y_true,y_delta,input_t_test,input_c_test,input_a_test,input_y_test,y_true_test,y_delta_test 



input_t,input_c,input_a,input_y,y_true,y_delta,input_t_test,input_c_test,input_a_test,input_y_test,y_true_test,y_delta_test = splitData(moniData, x_name)
  
data_before = np.concatenate([input_t.values.reshape(len(input_t),1),input_c],axis=1)

t_dim = 1
c_dim = x_dim
y_dim = 1
t_intermediate_dim = 3
intermediate_dim = round(x_dim/4 + 2)
rep_dim = round(x_dim/2 + 2)
batch_size = 800

epochs = 400


# Size of noise vector
# noise_dim = 1

def get_discriminator_model():
    t_input = layers.Input(shape = (t_dim,))
    x_random_input = layers.Input(shape = (rep_dim,))
    
    ## t and x's Concatenate, othewile t_information and x_information Concatenate
    is_indep_x = layers.Concatenate()([t_input,x_random_input])
    
    is_true_ = layers.Dense(intermediate_dim,activation = 'elu')(is_indep_x)
    is_true_ = layers.Dense(intermediate_dim,activation = 'elu')(is_true_)
    # kernel_initializer=initializers.random_normal(),
    # bias_initializer=initializers.random_normal()
    
    # is_true = layers.Dense(1,
    #               kernel_initializer=initializers.random_normal(),
    #               bias_initializer=initializers.random_normal())(is_true_)
    
    is_true = layers.Dense(1)(is_true_)
    
    d_model = keras.models.Model([t_input,x_random_input], is_true, name="discriminator")
    return d_model


"""
## Create the generator
"""
# x = layers.BatchNormalization()(x)

def get_generator_model():
    # noise_u = layers.Input(shape=(noise_dim,))
    c = layers.Input(shape=(c_dim,))
 
    x_rep_ = layers.Dense(intermediate_dim * 2, activation='elu')(c)
    x_rep_ = layers.Dense(intermediate_dim, activation='elu')(x_rep_)
    x_rep_ = layers.Dense(intermediate_dim, activation='elu')(x_rep_)
    x_rep = layers.Dense(rep_dim,activation='elu', name = 'x_rep')(x_rep_)
    
    g_model = keras.models.Model(c, x_rep, name="generator")
    return g_model


def generalizationSCM():
    # noise_u = layers.Input(shape=(noise_dim,))
    t = layers.Input(shape=(t_dim,))
    x_rep = layers.Input(shape=(rep_dim,))
    a_rep = layers.Input(shape=(a_dim,))
    
    x_rep_int_ = layers.Dense(round(1.5 * rep_dim) , activation='elu',kernel_regularizer = 'l1')(x_rep)
    x_rep_int_ = layers.Dense(round(rep_dim) , activation='elu',kernel_regularizer = 'l1')(x_rep_int_)
    x_rep_int = layers.Dense(round(0.5 * rep_dim) , activation='elu',kernel_regularizer = 'l1')(x_rep_int_)
    
    a_rep_int_ = layers.Dense(round(1.5 * a_dim) , activation='elu',kernel_regularizer = 'l1')(a_rep)
    a_rep_int_ = layers.Dense(round(rep_dim) , activation='elu',kernel_regularizer = 'l1')(a_rep_int_)
    a_rep_int = layers.Dense(round(0.5 * a_dim) , activation='elu',kernel_regularizer = 'l1')(a_rep_int_)
    
    # 
    t_int_ = layers.Dense(3 , activation='elu')(t)
    t_int = layers.Dense(2 , activation='elu')(t_int_)
    
    y_x = layers.Concatenate()([x_rep_int,a_rep_int,t_int])
    
    y_ = layers.Dense(3 , activation='elu',kernel_regularizer = 'l1')(y_x)
    y_ = layers.Dense(1 , activation='elu',kernel_regularizer = 'l1')(y_)
    
    # y_ = layers.Dense(3)(y_)
    y = layers.Dense(1)(y_)
    
    # model
    GSCM = keras.models.Model([t,x_rep,a_rep], y, name="generator")
    return GSCM

# GSCM = generalizationSCM()

"""
## Create a WGAN-GP model
"""

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        GSCM,
        rep_dim,
        discriminator_extra_steps=6,
        generator_extra_steps=3,
        GSCM_extra_steps=5,
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
        # prediction
        real_t_ = real_data[0]
        c_ = real_data[1]
        a_ = real_data[2]
        
        #with tf.GradientTape() as tape:
        # Generate fake images from the latent vector
        generate_x = self.generator(c_)
        generate_y = self.GSCM([real_t_,generate_x,a_])
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
        a = real_data[0][2]
        y = real_data[0][3]
        
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
                cf_y_g = self.GSCM([real_t,gene_x_rep,a])
                
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
                cf_y = self.GSCM([real_t,gene_x_rep,a])
                # Calculate the generator loss
                GSCM_loss = self.factual_loss(cf_y,y)
                
            # Get the gradients w.r.t the generator loss
            GSCM_gradient = tape.gradient(GSCM_loss, self.GSCM.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.GSCM_optimizer.apply_gradients(
                zip(GSCM_gradient, self.GSCM.trainable_variables)
            )
            
        
        return {"d_loss": d_loss, "g_loss": g_loss,"GSCM_loss": GSCM_loss}



"""
## Train the end-to-end model
"""


generator_optimizer = keras.optimizers.Adam() # learning_rate = 0.005
discriminator_optimizer = keras.optimizers.Adam()
GSCM_optimizer = keras.optimizers.Adam(learning_rate = 0.001)

def discriminator_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return fake_loss - real_loss

# Define the loss functions to be used for generator
def generator_loss(gen_img_logits):
    return -tf.reduce_mean(gen_img_logits)

from tensorflow.keras.losses import mse
def factual_loss(fac_y, y):
    return tf.reduce_mean(mse(fac_y,y))

train_RMSE_MTEF = []
test_RMSE_MTEF = []
train_RMSE_MSE = []
test_RMSE_MSE = []
for i in range(100):
    d_model = get_discriminator_model()
    g_model = get_generator_model()
    GSCM = generalizationSCM()
    # Get the wgan model
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        GSCM=GSCM,
        rep_dim=rep_dim,
    )
    
    # Compile the wgan model
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        GSCM_optimizer=GSCM_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
        factual_loss=factual_loss
    )
    
    # Start training
    history = wgan.fit([input_t, input_c,input_a,input_y],batch_size=batch_size, epochs=epochs, verbose = 0)
    
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
    
    predict = wgan.predict([input_t, input_c,input_a])
    
    x_reo_ = predict[0]
    data_later = np.concatenate([input_t.values.reshape(len(input_t),1),x_reo_],axis=1)

    later_columns = ['x_{}'.format(i) for i in range(rep_dim)]
    later_columns.insert(0,'t')
    
    data_later_d = pd.DataFrame(data_later,columns = later_columns)
    
    GSCM_predict = predict[1]
    
    plt.plot(y_true[:50])
    plt.plot(GSCM_predict[:50])
    plt.legend(['True_y', 'predict_y'], loc='upper left')
    plt.show()
    
    ## M
    true_MTEF = (y_true - y_delta) / 0.1
    
    GSCM_predict_delta = wgan.predict([input_t - 0.1, input_c, input_a])[1]
    predict_MTEF = (GSCM_predict - GSCM_predict_delta) / 0.1
    
    print(f"RMSE of MTEF:{np.sqrt(mean_squared_error(predict_MTEF, true_MTEF))}")
    train_RMSE_MTEF.append(round(np.sqrt(mean_squared_error(predict_MTEF, true_MTEF)),4))
    train_RMSE_MSE.append(round(np.sqrt(mean_squared_error(GSCM_predict, y_true)),4))
    
    predict_test = wgan.predict([input_t_test, input_c_test, input_a_test])
    
    x_reo_test = predict_test[0]
    GSCM_predict_test = predict_test[1]
   
    true_MTEF_test = (y_true_test - y_delta_test) / 0.1
    
    GSCM_predict_test_delta_ = wgan.predict([input_t_test - 0.1, input_c_test, input_a_test])[1]
    predict_MTEF_test = (GSCM_predict_test - GSCM_predict_test_delta_) / 0.1
    
    print(f"RMSE of MTEF:{np.sqrt(mean_squared_error(predict_MTEF_test, true_MTEF_test))}")
    test_RMSE_MTEF.append(round(np.sqrt(mean_squared_error(predict_MTEF_test, true_MTEF_test)),4))
    test_RMSE_MSE.append(round(np.sqrt(mean_squared_error(GSCM_predict_test, y_true_test)),4))
    
train_RMSE_MTEF.remove(max(train_RMSE_MTEF))
train_RMSE_MTEF.remove(min(train_RMSE_MTEF))

test_RMSE_MTEF.remove(max(test_RMSE_MTEF))
test_RMSE_MTEF.remove(min(test_RMSE_MTEF))

train_RMSE_MSE.remove(max(train_RMSE_MSE))
train_RMSE_MSE.remove(min(train_RMSE_MSE))

test_RMSE_MSE.remove(max(test_RMSE_MSE))
test_RMSE_MSE.remove(min(test_RMSE_MSE))

print('Mean Value of RMSE of MTEF on Train Set:',np.mean(train_RMSE_MTEF))
print('Mean Value of RMSE of MTEF on Test Set:',np.mean(test_RMSE_MTEF))

print('Std Value RMSE of MTEF on Train Set:',np.std(train_RMSE_MTEF))
print('Std Value RMSE of MTEF on Test Set:',np.std(test_RMSE_MTEF))

print('Mean Value of RMSE of MSE on Train Set:',np.mean(train_RMSE_MSE))
print('Mean Value of RMSE of MSE on Test Set:',np.mean(test_RMSE_MSE))

print('Std Value RMSE of MSE on Train Set:',np.std(test_RMSE_MSE))
print('Std Value RMSE of MSE on Test Set:',np.std(test_RMSE_MSE))