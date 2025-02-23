import joblib as jl
import enum
import warnings
import tensorflow as tf
import numpy as np
warnings.filterwarnings("ignore")

manure_classifier_model_info = jl.load('models/manure_classifier.joblib')
rf_model_info = jl.load('models/rf_model.joblib')
teanet_model_info = jl.load('models/teanet_model.joblib')
tearesnet_model_info = jl.load('models/tearesnet_model.joblib')

class Options(enum.Enum):
    SVM = "SVM"
    RANDOM_FOREST = "RF"
    TEANET = "TN"
    TEARESNET = "TRN"

def convert_to_pseudo_image(Spectra_data):
  Spectra_data = tf.convert_to_tensor(Spectra_data)
  Spectra_data = tf.expand_dims(Spectra_data,axis=2)
  Spectra_data = tf.expand_dims(Spectra_data,axis=3)
  Spectra_data = tf.reshape(Spectra_data[:,:675,:,:],(Spectra_data.shape[0],15,15,3))
  return Spectra_data

def pred(val, opt:str):
    model_info = None
    if opt == Options.SVM.value:
        model_info = manure_classifier_model_info
    elif opt == Options.RANDOM_FOREST.value:
        model_info = rf_model_info
    elif opt == Options.TEANET.value:
        model_info = teanet_model_info
    elif opt == Options.TEARESNET.value:
        model_info = tearesnet_model_info
    else:
        model_info = teanet_model_info
    
    model = model_info['model']
    label_encoder = model_info['label_encoder']
    if opt in [Options.TEANET.value, Options.TEARESNET.value]:
        _val = np.array([val])
        _val.reshape(1, -1)
        _image_val = convert_to_pseudo_image(_val)
        val_pred = model.predict(_image_val)
        y_pred_classes = val_pred.argmax(axis=1)
        return label_encoder.inverse_transform(y_pred_classes)
    else:
        val_pred = model.predict([val])
        return label_encoder.inverse_transform(val_pred)
