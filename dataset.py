import random
import os
import numpy as np
import librosa
from torch.utils import data


class AudioDataset(data.Dataset):
    def __init__(self, base_path: str, limit_num_samples: int = None,
                 fixed_length: int = 22050, train_val_test_split: tuple = (0.8, 0.1, 0.1),
                 mode: str = 'train', seed: int = None, normalize: bool = True):
        """
        :param base_path: path to the data folder;
        :param limit_num_samples: limits the number of samples of each class
            to the number set;
        :param fixed_length: number of samples to pad/strip each audio to;
        :param train_val_test_split: fraction of the data, reserved for the
            train/val/test set correspondingly;
        :param mode: one of ["train", "val", "test"];
        :param seed: random seed to fix the way data is shuffled;
        :param normalize: if True, audio is normalized to the range of [-1, 1];
        """

        self._base_path = base_path
        self._limit = limit_num_samples if limit_num_samples else np.Infinity
        self._fixed_length = fixed_length
        self._normalize = normalize

        self.wav_to_label = self._parse_data()
        self.wavlist = list(self.wav_to_label.keys())

        if seed:
            random.seed(seed)
        random.shuffle(self.wavlist)

        # selecting the part of filelist depending on the dataset mode;
        # randomly selecting chunks may lead to unballanced sets, especially
        # in case of small val/test subsets.
        # TODO: think about some balancing procedure
        if mode == 'train':
            self.wavlist = self.wavlist[:round(len(self.wavlist) * train_val_test_split[0])]
        elif mode == 'val':
            i_from = round(len(self.wavlist) * train_val_test_split[0])
            i_to = round(len(self.wavlist) * (train_val_test_split[0] + train_val_test_split[1]))
            self.wavlist = self.wavlist[i_from:i_to]
        elif mode == 'test':
            self.wavlist = self.wavlist[-round(len(self.wavlist) * train_val_test_split[2]):]

        self._label_to_id = {
            'down': 0,
            'go': 1,
            'left': 2,
            'no': 3,
            'off': 4,
            'on': 5,
            'right': 6,
            'stop': 7,
            'up': 8,
            'yes': 9
        }

    def _parse_data(self) -> dict:
        wav_to_label = {}

        for label in os.listdir(self._base_path):
            dirpath = os.path.join(self._base_path, label)
            if os.path.isdir(dirpath):
                for i, file_ in enumerate(os.listdir(dirpath)):
                    if file_[-4:] == '.wav' and i < self._limit:
                        full_path = os.path.join(dirpath, file_)
                        wav_to_label[full_path] = label

        return wav_to_label

    @staticmethod
    def _zero_pad_sequence(sequence: np.ndarray, pad_to: int) -> np.ndarray:
        padded_sequence = np.zeros(pad_to)
        padded_sequence[:len(sequence)] = sequence

        return padded_sequence

    def __getitem__(self, index: int) -> dict:
        wavpath = self.wavlist[index]
        label = self.wav_to_label[wavpath]
        audio, sr = librosa.load(wavpath)

        if self._normalize:
            audio = librosa.util.normalize(audio)

        if self._fixed_length:
            if len(audio) < self._fixed_length:
                audio = self._zero_pad_sequence(audio, self._fixed_length)
            else:
                audio = audio[:self._fixed_length]

        return {
            'name': wavpath,
            'audio': audio,
            'label': self._label_to_id[label],
            'mfcc': librosa.feature.mfcc(audio)
        }

    def __len__(self) -> int:
        return len(self.wavlist)
