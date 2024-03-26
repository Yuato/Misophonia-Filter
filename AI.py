import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

CAPUCHIN_FILE = os.path.join('data', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')

print(CAPUCHIN_FILE)
print(NOT_CAPUCHIN_FILE)

file_contents = tf.io.read_file(CAPUCHIN_FILE)

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

wave = load_wav_16k_mono(CAPUCHIN_FILE)
nwave = load_wav_16k_mono(NOT_CAPUCHIN_FILE)

#plots the audio files into a visual wave
'''
plt.plot(wave)
plt.plot(nwave)
plt.show()
'''

POS = os.path.join('data', 'Parsed_Capuchinbird_Clips')
NEG = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips')

pos = tf.data.Dataset.list_files(POS+'\*.wav')
pos.as_numpy_iterator().next()
neg = tf.data.Dataset.list_files(NEG+'\*.wav')

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))#returns one binary flag for positive examples
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))#returns zeros for negative examples
data = positives.concatenate(negatives)#puts the two data sets together

print(data.shuffle(1000).as_numpy_iterator().next())

lengths = []
for file in os.listdir(os.path.join('data', 'Parsed_Capuchinbird_Clips')):
    tensor_wave = load_wav_16k_mono(os.path.join('data', 'Parsed_Capuchinbird_Clips', file))
    lengths.append(len(tensor_wave))

print(tf.math.reduce_mean(lengths))
print(tf.math.reduce_min(lengths))#prints minimum wave length
tf.math.reduce_max(lengths)#prints maximum wave length

def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)#adds zeros at the start of clips that are less than 48000(can be changed) in size
    wav = tf.concat([zero_padding, wav],0)#adds the padding with the wav
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)#all positive examples
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


