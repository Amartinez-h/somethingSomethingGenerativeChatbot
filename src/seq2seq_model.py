import numpy as np
import tensorflow as tf
import time

class seq2seq_model():
    
    def __init__(self, inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words,
                 encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
        
        encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, 
                                                                  answers_num_words+1,
                                                                  encoder_embedding_size,
                                                                  initializer=tf.random_uniform_initializer(0, 1))
        
        encoder_state = self.encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
        
        preprocessed_targets = self.preprocess_targets(targets, questionswords2int, batch_size)
        
        decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1, decoder_embedding_size], 0, 1))
        
        decoder_embeddings_input = tf.nn.embedding_lookup(decoder_embedding_matrix, preprocessed_targets)
        
        self.training_predictions, self.test_predictions = decoder_rnn(decoder_embedded_input,
                                                             decoder_embeddings_matrix,
                                                             encoder_state,
                                                             questions_num_words,
                                                             sequence_length,
                                                             questionswords2int,
                                                             keep_prob,
                                                             batch_size)
    
    def model_inputs(self):
        ''' Creates placeholders for the inputs'''
        inputs = tf.placerholder(tf.int32, [None, None], name = 'input')
        targets = tf.placerholder(tf.int32, [None, None], name = 'target')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        return inputs, targets, lr, keep_prob
    
    
    def preprocess_targets(self, targets, word2int, batch_size):
        '''Preprocess the targets'''
        # First, we append <SOS> token to the end of every sentence
        left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
        
        # To keep the sentences the same size, the last token is not used
        right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
        
        # The sentences minus the last token are appended to the <SOS> tokens
        prepocessed_targets = tf.concat([left_side, right_side], axis=1, name="preprocessed_targets")
        
        return prepocessed_targets
    
    
    def encoder_rnn(self, rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
        ''' Encoder RNN layer creation'''
        # Basic LSTM cell 
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        
        # Dropout addition to the input of our cells
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        
        # Encoding layout
        encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        
        # Encoded state to be returned (This might make more sense in the future)
        # encoder_output, encoder_state
        _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                           cell_bw = encoder_cell, 
                                                           sequence_length = sequence_length,
                                                           inputs = rnn_inputs,
                                                           dtype = tf.float32)
        
        return encoder_state
    
    
    def decode_training_set(self, encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, 
                             output_function, keep_prob, batch_size):
        '''Training set decoding'''
        # Array to store attention vectors
        attention_states = tf.zeros([batch_size, 1, decoder_cell.ouput_size])
        
        # Attention related variables obtention
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                        attention_option = 'bahdanau', 
                                                                                                                                        num_units = decoder_cell.output_size)
        
        # Function to train the decorder procedure
        training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], 
                                                                                  attention_keys, 
                                                                                  attention_values,
                                                                                  attention_score_function, 
                                                                                  attention_construct_function, 
                                                                                  name = 'attn_dec_train')
        
        # Decoder obtention
        # decoder_output, decoder_final_state, decoder_final_context_state
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                      training_decoder_function, 
                                                                      decoder_embedded_input, 
                                                                      sequence_length, 
                                                                      scope = decoding_scope)
        
        # Addition of a dropout factor to the decoder
        decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
        
        return output_function(decoder_output_dropout)
    
    
    def decode_test_set(self, encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, max_length, num_words,
                             sequence_length, decoding_scope, output_function, keep_prob, batch_size,):
        
        # Array to store attention vectors
        attention_states = tf.zeros([batch_size, 1, decoder_cell.ouput_size])
        
        # Attention related variables obtention
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                        attention_option = 'bahdanau',
                                                                                                                                        num_units = decoder_cell.output_size)
        
        # Function to test the decoder 
        test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function, 
                                                                                  encoder_state[0], 
                                                                                  attention_keys, 
                                                                                  attention_values, 
                                                                                  attention_score_function, 
                                                                                  attention_construct_function,  
                                                                                  decoder_embeddings_matrix, 
                                                                                  sos_id, 
                                                                                  eos_id, 
                                                                                  max_length,
                                                                                  num_words,
                                                                                  name = 'attn_dec_inf')
        
        # Test predictions retrieval
        # decoder_output, decoder_final_state, decoder_final_contexrt_state
        test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                        test_decoder_function, 
                                                                        scope = decoding_scope)
        
        return test_predictions
    
    
    def decoder_rnn(self, decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size,
                    num_layers, word2int, keep_prob, batch_size):
        with tf.variable_scope("decoding") as decoding_scope:
            # Basic LSTM cell 
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            
            # Dropout addition to the input of our cells
            lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
            
            # Encoding layout
            decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
            
            weights = tf.truncated_normal_initializer(stddev=0.1)
            
            biases = tf.zeros_initializer()
            
            output_function = lambda x: tf.contrib.layers.fully_connected(x, 
                                                                          num_words,
                                                                          None, 
                                                                          scope = decoding_scope,
                                                                          weights_initializer=weights,
                                                                          biases_initializer=biases)
            
            training_predictions = self.decode_training_set(encoder_state,
                                                            decoder_cell,
                                                            decoder_embedded_input,
                                                            sequence_length,
                                                            decoding_scope, 
                                                            output_function,
                                                            keep_prob,
                                                            batch_size)
            
            # We indicate to the scope that we want to reuse variables 
            decoding_scope.reuse_variables()
            
            test_predictions = self.decode_test_set(encoder_state,
                                                    decoder_cell,
                                                    decoder_embeddings_matrix,
                                                    word2int['<SOS>'],
                                                    word2int['<EOS>'],
                                                    sequence_length-1,
                                                    num_words,
                                                    decoding_scope,
                                                    output_function,
                                                    keep_prob,
                                                    batch_size)
            
            return training_predictions, test_predictions
        
            
            
            