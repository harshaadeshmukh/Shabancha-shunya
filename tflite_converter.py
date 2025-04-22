import tensorflow as tf

model = tf.keras.models.load_model('marathi_sign_language_model.h5')
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)


tflite_model = tflite_converter.convert()
open("msl_model.tflite", "wb").write(tflite_model)

