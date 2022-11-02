import librosa
import pickle
import numpy as np
from model.dataloader import Audio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class Preprocessor:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    #TODO("should be corrected due to bad normalization method")
    def __get_audio_features(self):
        features = []
        for i in list(self.X):
            x, sr = librosa.load(i)
            mfcc = librosa.feature.mfcc(y=x, sr=1, n_mfcc=40)
            mel = librosa.feature.melspectrogram(y=x, sr=sr)
            S = np.abs(librosa.stft(x))
            spec = librosa.feature.spectral_contrast(S=S, sr=sr)
            y = librosa.effects.harmonic(x)
            tonne = librosa.feature.tonnetz(y=y, sr=sr)
            feature_list = np.concatenate((mfcc, mel, spec, tonne), axis=0)
            features.append(self.normalize_features(feature_list))
        return features

    def normalize_features(self, original_list):
        return preprocessing.normalize(original_list, axis=1)

    def train_test_val_split(self):
        normalized_X = self.__get_audio_features()
        X_train, X_test, y_train, y_test = train_test_split(normalized_X, self.y, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
        return X_train, X_val, X_test, y_train, y_test, y_val


if __name__ == "__main__":
    audio1_database = Audio("../../databases/EmotionRecognition_joined/RAVDESS_augumented").load_data()
    audio2_database = Audio("../../databases/EmotionRecognition_joined/RML").load_data()
    audio3_database = Audio("../../databases/EmotionRecognition_joined/SAVEE").load_data()
    y = []
    X = []

    for i in audio1_database + audio2_database + audio3_database:
        y.append(i.label)
        X.append(i.filepath)

    X_train, X_val, X_test, y_train, y_test, y_val = Preprocessor(X, y).train_test_val_split()
    pickle_dict = {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_test": y_test, "y_val": y_val}

    with open('../audio_data.pickle', 'wb') as handle:
        pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
