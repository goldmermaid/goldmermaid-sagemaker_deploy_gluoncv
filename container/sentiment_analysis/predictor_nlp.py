
# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback
import flask

import math
import numpy as np
import mxnet as mx
import gluonnlp as nlp


prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
CONTENT_TYPE_JSON = 'application/json'
CONTENT_TYPE_CSV = 'text/csv'


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = {}                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls, 
                  model_name='bert_12_768_12', 
                  dataset_name='book_corpus_wiki_en_uncased', 
                  ctx = mx.gpu(0),
                  pretrained=True):
        """Get the model object for this instance, loading it if it's not already loaded."""
        
        
        bert, vocabulary = nlp.model.get_model(model_name, # the 12-layer BERT Base model
                                            dataset_name=dataset_name,
                                            # use pre-trained weights
                                            pretrained=pretrained, ctx=ctx,
                                            # decoder and classifier are for pre-training only
                                            use_decoder=False, use_classifier=False)
        net = nlp.model.BERTClassifier(bert, num_classes=2)
#         net.classifier.initialize(ctx=ctx)  # only initialize the classification layer from scratch
#         net.hybridize()
        return net, vocabulary
        


    @classmethod
    def predict_sentiment(net, vocabulary, input_sentence, ctx=mx.gpu(0)):
        """
        For the input sentence, predict its sentiment.

        Args:
        - net : trained BERT model with parameters
        - vocabulary : a dict that save each token (key)'s related ID (value)
        - input_sentence : a string on which to do the predictions. 
        """
        
        ctx = ctx[0] if isinstance(ctx, list) else ctx
        max_len = 128
        padding_id = vocabulary[vocabulary.padding_token]
        bert_tokenizer = nlp.data.BERTTokenizer(vocabulary)
        inputs = mx.nd.array([vocabulary[['[CLS]'] + bert_tokenizer(sentence) + ['SEP']]], ctx=ctx)
        seq_len = mx.nd.array([inputs.shape[1]], ctx=ctx)
        token_types = mx.nd.zeros_like(inputs)
        out = net(inputs, token_types, seq_len)
        label = mx.nd.argmax(out, axis=1)
        return 'positive' if label.asscalar() == 1 else 'negative'

    
# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data. 
    In this sample server, we take the input sentence and apply prediction.
    """
    
    data = None
    if content_type == CONTENT_TYPE_JSON:
        payload = flask.request.data
        data = json.loads(payload.decode("utf-8"))
    except:
        return flask.Response(response='Cannot decode the input', status=415, mimetype='text/plain')

    # Do the prediction
    net, vocabulary = ScoringService.get_model()
    result = ScoringService.predict(net, vocabulary, data)

    return flask.Response(response=result, status=200, mimetype='text/plain')
