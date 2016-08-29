#!/usr/bin/env python

from keras.layers import Recurrent
from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import InputSpec
import numpy as np

class HHMMLayer(Recurrent):
    def __init__(self, embed_dim, syn_dim,
         init='glorot_uniform', inner_init='orthogonal',
         forget_bias_init='one', activation='tanh',
         inner_activation='relu', **kwargs):

        self.embed_dim = embed_dim
        self.syn_dim = syn_dim
        self.output_dim = syn_dim * 2 + embed_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.bool_act = activations.get('sigmoid')
        super(HHMMLayer, self).__init__(**kwargs)
        
    ## Initialize everything:
    """ this is where you will define your weights. Trainable weights should be added to 
    the list self.trainable_weights. Other attributes of note are: 
    self.non_trainable_weights (list) and self.updates (list of update tuples 
    (tensor, new_tensor)). For an example of how to use non_trainable_weights and updates,
    see the code for the BatchNormalization layer."""
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None] * 3

        self.input_dim = input_shape[2]
        self.W_f = self.init( (self.syn_dim+self.embed_dim, 1), name="W_f" )
        self.b_f = K.zeros( (1,), name="b_f" )
        
        self.W_j_p = self.init( (self.syn_dim+self.embed_dim, 1), name="W_j_p")
        self.b_j_p = K.zeros( (1,), name="b_j_p")
        
        self.W_j_m = self.init( (self.syn_dim, 1), name="W_j_m")
        self.b_j_m = K.zeros( (1,), name="b_j_m")
        
        self.W_a_mm = self.init( (self.syn_dim, self.syn_dim), name="W_a_mm")
        self.b_a_mm = K.zeros( (self.syn_dim,), name="b_a_mm")
        
        self.W_a_pm = self.init( (self.syn_dim+self.embed_dim,self.syn_dim), name="W_a_pm")
        self.b_a_pm = K.zeros( (self.syn_dim,), name="b_a_pm")
        
        self.W_b_p = self.init( (self.syn_dim+self.embed_dim, self.syn_dim), name="W_b_p")
        self.b_b_p = K.zeros( (self.syn_dim,), name="b_b_p")
        
        self.W_b_m = self.init( (self.syn_dim*2, self.syn_dim),name="W_b_m")
        self.b_b_m = K.zeros( (self.syn_dim,), name="b_b_m")
        
        self.trainable_weights = [self.W_f, self.b_f,
                                  self.W_j_p, self.b_j_p, self.W_j_m, self.b_j_m,
                                  self.W_a_mm, self.b_a_mm, self.W_a_pm, self.b_a_pm,
                                  self.W_b_p, self.b_b_p, self.W_b_m, self.b_b_m]
            
    ## Propagate information one step
    ## The Recurrent.call() method will call this
    def step(self, x, states):
        prev_act = states[0]
        prev_awa = states[1]
        prev_token = states[2]
        
        ## Concatenate vectors for awa variable and word embedding variable:
        ## (1 x [embed_dim+syn_dim])
        ## then dot by f weights:
        ## ( [embed_dim+syn_dim] x syn_dim )
        awa_em = K.concatenate( (prev_awa, prev_token) )
        f = self.bool_act( K.dot( awa_em, self.W_f ) + self.b_f )
        
        j = self.bool_act( 
                (f) * ( K.dot( awa_em, self.W_j_p ) + self.b_j_p ) + (1-f) * ( K.dot( prev_act, self.W_j_m ) + self.b_j_m ) )
        
        act_awa = K.concatenate( (prev_act, prev_awa) )
        act = f * j * prev_act + (1-f) * (1-j) * self.activation( K.dot( prev_act, self.W_a_mm ) + self.b_a_mm ) + f * (1-j) * self.activation( K.dot( act_awa, self.W_a_pm ) + self.b_a_pm)
        
        act_pair = K.concatenate( (prev_act, act) )
        awa = j * self.activation( K.dot( awa_em, self.W_b_p ) + self.b_b_p ) + (1 - j) * self.activation( K.dot( act_pair, self.W_b_m ) + self.b_b_m )
        
        #output = np.array( [f, j] )
        output = K.concatenate( (f, j) )
        return output, [act, awa, x]
        
    ## Return config variables (LSTM config shown in comments below)
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__}
        base_config = super(HHMMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
