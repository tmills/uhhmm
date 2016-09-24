#!/usr/bin/env python

from keras.layers import Recurrent
from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import InputSpec
import numpy as np

class HHMMLayer(Recurrent):
    def __init__(self, embed_dim, syn_dim,
         init='glorot_uniform', 
         activation='tanh', **kwargs):

        self.embed_dim = embed_dim
        self.syn_dim = syn_dim
        self.output_dim = 2 + self.syn_dim * 2
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        super(HHMMLayer, self).__init__(**kwargs)
        
    ## Initialize everything:
    """ this is where you will define your weights. Trainable weights should be added to 
    the list self.trainable_weights. Other attributes of note are: 
    self.non_trainable_weights (list) and self.updates (list of update tuples 
    (tensor, new_tensor)). For an example of how to use non_trainable_weights and updates,
    see the code for the BatchNormalization layer."""
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        
        self.bool_act = activations.get('hard_sigmoid')
        self.zero = K.zeros((1,))
        self.one = K.zeros((1,)) + 1
        self.threshold = K.zeros(1) + 0.5

        if self.stateful:
            self.reset_states()
        else:
            # initial states: f, j, a, b, x
            self.states = [None] * 3

        self.input_dim = input_shape[2]
        self.W_f = self.init( (self.syn_dim+self.embed_dim, 1), name="W_f" )
        self.b_f = self.init( (1,), name="b_f" )
        
        self.W_j_p = self.init( (self.syn_dim+self.embed_dim, 1), name="W_j_p")
        self.b_j_p = self.init( (1,) ,name="b_j_p" )
        
        self.W_j_m = self.init( (self.syn_dim, 1), name="W_j_m")
        self.b_j_m = self.init( (1,), name="b_j_m")
        
        self.W_a_mm = self.init( (self.syn_dim + self.embed_dim, self.syn_dim), name="W_a_mm")
        self.b_a_mm = self.init( (self.syn_dim,), name="b_a_mm")
        
        self.W_a_pm = self.init( (self.syn_dim*2 + self.embed_dim,self.syn_dim), name="W_a_pm")
        self.b_a_pm = self.init( (self.syn_dim,), name="b_a_pm")
        
        self.W_b_p = self.init( (self.syn_dim+self.embed_dim*2, self.syn_dim), name="W_b_p")
        self.b_b_p = self.init( (self.syn_dim,), name="b_b_p")
        
        self.W_b_m = self.init( (self.syn_dim*2+self.embed_dim, self.syn_dim),name="W_b_m")
        self.b_b_m = self.init( (self.syn_dim,), name="b_b_m")

        self.trainable_weights = [self.W_f, self.b_f,
                                  self.W_j_p, self.b_j_p, self.W_j_m, self.b_j_m,
                                  self.W_a_mm, self.b_a_mm, self.W_a_pm, self.b_a_pm,
                                  self.W_b_p, self.b_b_p, self.W_b_m, self.b_b_m]
    
    def get_initial_states(self, x):
        ## x has shape (samples, timesteps, input_dim)
        ## want initial states to have dimension:
        ## [ (samples, syn_dim), (samples, syn_dim), (samples, embed_dim) ]
        init_f = K.zeros_like(x) - 1    # (samples,1)
        init_f = K.squeeze(init_f, 2)
        init_f = K.squeeze(init_f, 1)
        init_f = K.reshape(init_f, (1, x.shape[0]) )
        
        init_j = K.zeros_like(x) - 1
        init_j = K.squeeze(init_j, 2)
        init_j = K.squeeze(init_j, 1)
        init_j = K.reshape(init_j, (1, x.shape[0]) )
        
#        init_j = K.zeros( (1, x.shape[0]) ) - 1    # (samples,1) 
        
        init_act = K.zeros_like(x)
        init_act = K.sum(init_act, axis=(1,2)) # (samples,)
        init_act = K.expand_dims(init_act) # (samples, 1)
        init_act = K.tile(init_act, [1, self.syn_dim]) # (samples, syn_dim)
        
        init_awa = K.zeros_like(init_act)
        
        init_embed = K.zeros_like( x )
        init_embed = K.sum(init_embed, axis=(1,2)) # (samples,)
        init_embed = K.expand_dims(init_embed) #(samples, 1)
        init_embed = K.tile(init_embed, [1, self.embed_dim])
        
        return [init_act, init_awa, init_embed] #, init_f, init_j]

    ## Propagate information one step
    ## The Recurrent.call() method will call this
    def step(self, x, states):
        #prev_f = states[0]
        #prev_j = states[1]
        prev_act = states[0]
        prev_awa = states[1]
        prev_token = states[2]
        #prev_f = states[3]
        #prev_j = states[4]
        
        ## Concatenate vectors for awa variable and word embedding variable:
        ## (1 x [embed_dim+syn_dim])
        ## then dot by f weights:
        ## ( [embed_dim+syn_dim] x syn_dim )
        awa_em = K.concatenate( (prev_awa, prev_token) )
        
        f = self.bool_act( K.dot( awa_em, self.W_f ) + self.b_f )
        if K.greater(f, self.threshold):
            f = f / f
        else:
            f = f * 0
        
        j = self.bool_act( 
                    (f) * ( K.dot( awa_em, self.W_j_p ) + self.b_j_p ) + (1-f) * ( K.dot( prev_act, self.W_j_m ) + self.b_j_m ) )
                    
        if K.greater(j, self.threshold):
            j = j / j 
        else:
            j = j * 0
            

        act_awa_x = K.concatenate( (prev_act, prev_awa, x) )
        prev_act_x = K.concatenate( (prev_act, x) )
        
        pp_term = K.tile(f * j, [1, self.syn_dim] ) * prev_act
        mm_term = K.tile( (1-f) * (1-j), [1, self.syn_dim])  * self.activation( K.dot( prev_act_x, self.W_a_mm ) + self.b_a_mm )
        pm_term = K.tile(f * (1-j), [1, self.syn_dim]) * self.activation( K.dot( act_awa_x, self.W_a_pm ) + self.b_a_pm)      
        
        act = pp_term + mm_term + pm_term      
        
        act_x_pair = K.concatenate( (prev_act, act, x) )
        awa_em_x = K.concatenate( (awa_em, x) )
        
        p_term = K.tile(j, [1, self.syn_dim]) * self.activation( K.dot( awa_em_x, self.W_b_p ) + self.b_b_p )
        m_term = K.tile((1 - j), [1, self.syn_dim]) * self.activation( K.dot( act_x_pair, self.W_b_m ) + self.b_b_m )
        awa = p_term + m_term
        
        output = K.concatenate( (f, j, act, awa) )
        return output, [act, awa, x] #, f, j]
        
    ## Return config variables (LSTM config shown in comments below)
    def get_config(self):
        config = {'embed_dim': self.embed_dim,
                  'syn_dim': self.syn_dim,
                  #'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__}
        base_config = super(HHMMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

