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
        self.output_dim = 2
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
            # initial states: f, j, a, b, x
#            self.states = [ K.zeros((int(input_shape[0]), int(self.syn_dim))),
#                            K.zeros((int(input_shape[0]), int(self.syn_dim))),
#                            K.zeros((int(input_shape[0]), int(self.embed_dim))) ]
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
        
        self.W_a_pm = self.init( (self.syn_dim*2,self.syn_dim), name="W_a_pm")
        self.b_a_pm = K.zeros( (self.syn_dim,), name="b_a_pm")
        
        self.W_b_p = self.init( (self.syn_dim+self.embed_dim, self.syn_dim), name="W_b_p")
        self.b_b_p = K.zeros( (self.syn_dim,), name="b_b_p")
        
        self.W_b_m = self.init( (self.syn_dim*2, self.syn_dim),name="W_b_m")
        self.b_b_m = K.zeros( (self.syn_dim,), name="b_b_m")
        
        self.trainable_weights = [self.W_f, self.b_f,
                                  self.W_j_p, self.b_j_p, self.W_j_m, self.b_j_m,
                                  self.W_a_mm, self.b_a_mm, self.W_a_pm, self.b_a_pm,
                                  self.W_b_p, self.b_b_p, self.W_b_m, self.b_b_m]
    
    def get_initial_states(self, x):
        ## x has shape (samples, timesteps, input_dim)
        ## want initial states to have dimension:
        ## [ (samples, syn_dim), (samples, syn_dim), (samples, embed_dim) ]
        init_act = K.zeros_like(x)
        init_act = K.sum(init_act, axis=(1,2)) # (samples,)
        init_act = K.expand_dims(init_act) # (samples, 1)
        init_act = K.tile(init_act, [1, self.syn_dim]) # (samples, syn_dim)
        
        init_awa = K.zeros_like(init_act)
        
        
        init_embed = K.zeros_like( x )
        init_embed = K.sum(init_embed, axis=(1,2)) # (samples,)
        init_embed = K.expand_dims(init_embed) #(samples, 1)
        init_embed = K.tile(init_embed, [1, self.embed_dim])
        
        return [init_act, init_awa, init_embed]

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
        
        pp_term = K.tile(f * j, [1, self.syn_dim] ) * prev_act
        mm_term = K.tile( (1-f) * (1-j), [1, self.syn_dim])  * self.activation( K.dot( prev_act, self.W_a_mm ) + self.b_a_mm )
        pm_term = K.tile(f * (1-j), [1, self.syn_dim]) * self.activation( K.dot( act_awa, self.W_a_pm ) + self.b_a_pm)
        
        act = pp_term + mm_term + pm_term
        
        act_pair = K.concatenate( (prev_act, act) )       
        p_term = K.tile(j, [1, self.syn_dim]) * self.activation( K.dot( awa_em, self.W_b_p ) + self.b_b_p )
        m_term = K.tile((1 - j), [1, self.syn_dim]) * self.activation( K.dot( act_pair, self.W_b_m ) + self.b_b_m )
        awa = p_term + m_term
        
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
