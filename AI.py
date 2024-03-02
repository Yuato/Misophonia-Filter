import os
from matplotlib import pyplot as pyplot
import tensorflow as tf
import tensorflow_io as tfio

CAPUCHIN_FILE = os.path.join('data', 'ParsedCapuchinebird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forst-0.wav')

print(CAPUCHIN_FILE)

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

POS = os.path.join('data', 'Parsed_Capuchinbird_Clips')
NEG = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips')

pos = tf.data.Dataset.list_files(POS+'\*.wav')
neg = tf.data.Dataset.list_files(NEG+'\*.wav')

pos.as_numpy_iterator().next()

positives = tf.data.Dataset.sip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.sip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
positives.concatenate(negatives)

len(pos)
