import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#function to extract the mfcc, chroma, and mel features from a sound file.

def extract_feature(file_name,mfcc, chroma, mel):
    #opening the sound file
    with soundfile.SoundFile(file_name) as sound_file: 
        #extracting the features from the sound file
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        #If chroma is True, get the Short-Time Fourier Transform of X
        if chroma:
            stft = np.abs(librosa.stft(X))
        #result is an empty numpy array
        result = np.array([])    
            
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        #hstack() stacks arrays in sequence horizontally
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result
            
# define a dictionary to a dictionary the emotions available in the RAVDESS dataset, and a list to hold those we want.       
emotions = {
    "01":"neutral",
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}

# Emotions we are interested in
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

def load_data(test_size=0.2):
    # x, y is are lists that will hold the features and emotions.
    x, y = [], []
    # We gonna parse the files in the current directory. 
    # The glob() function from the glob module to get all the pathnames for the sound files in our dataset. 
    for file in glob.glob("speech-emotion-recognition-ravdess-data/Actor_*/*.wav"):
        #need to get the file name so we can retrieve the emotion. 
        file_name = os.path.basename(file)
        #retrieving the emotion
        emotion = emotions[file_name.split("-")[2]]
 #emotion not is observed_emotions we will look again through the file names
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
                      
        x.append(feature)
        y.append(emotion)
            
    return train_test_split(np.array(x), y, test_size=test_size, random_state = 9)

#split the data set
x_train, x_test, y_train, y_test = load_data(test_size = 0.25)

#checking the shape of the training and the testing set.
print((x_train.shape[0], x_test.shape[0]))

#Get the number of the features.
print(f'Features extracted: {x_train.shape[1]}')

# Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#fit and train the model
model.fit(x_train, y_train)

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of our model
accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy: {}%".format(accuracy*100))