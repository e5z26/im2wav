import librosa
import math
import numpy as np
import models.utils.dist_adapter as dist
from torch.utils.data import Dataset
from models.utils.dist_utils import print_all
from models.utils.io import get_duration_sec, load_audio
import pickle
from pathlib import Path
import os
class FilesAudioDataset(Dataset):
    def __init__(self, hps):
        super().__init__()
        self.sr = hps.sr
        self.channels = hps.channels
        self.min_duration = hps.min_duration or math.ceil(hps.sample_length / hps.sr)
        if self.min_duration < hps.sample_length / hps.sr:
            print(f"changing min_durationfrom {self.min_duration} {math.ceil(hps.sample_length / hps.sr)}")
            self.min_duration =  math.ceil(hps.sample_length / hps.sr)

        self.max_duration = hps.max_duration or math.inf
        self.sample_length = hps.sample_length
        assert hps.sample_length / hps.sr < self.min_duration, f'Sample length {hps.sample_length} per sr {hps.sr} ({hps.sample_length / hps.sr:.2f}) should be shorter than min duration {self.min_duration}'
        self.aug_shift = hps.aug_shift
        self.labels = hps.labels
        self.init_dataset(hps)

    def filter(self, files, durations, requiredTypes=None):
        if requiredTypes is not None:
            with open("data/audio2tags_dev.pickle", 'rb') as handle:
                audio2tags = pickle.load(handle)
        # Remove files too short or too long
        keep = []
        filter = []
        for i in range(len(files)):
            fname = Path(files[i]).stem
            if not self.image_clip_emb:
                fname = int(fname)
            if durations[i] / self.sr < self.min_duration:
                filter.append(i)
                continue
            if durations[i] / self.sr >= self.max_duration:
                filter.append(i)
                continue
            if requiredTypes is not None:
                audio_tags = audio2tags[fname]
                if not isinstance(audio_tags, list):
                    filter.append(i)
                    continue
                audio_tags = set(audio_tags)
                if len(requiredTypes.intersection(audio_tags)) == 0:
                    filter.append(i)
                    continue
            if self.clip_emb:
                # For image CLIP embeddings, check if name exists in loaded dictionary
                if self.image_clip_emb:
                    if fname not in self.file2CLIP:
                        filter.append(i)
                        continue
                else:
                    # For video CLIP embeddings, check for individual pickle files
                    if not os.path.exists(os.path.join(self.CLIP_path, f'{fname}.pickle')):
                        filter.append(i)
                        continue
            if durations[i] == -1:
                filter.append(i)
                continue
            keep.append(i)
        print_all(f'self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}')
        print_all(f"Keeping {len(keep)} of {len(files)} files")
        self.files = [files[i] for i in keep]
        self.durations = [int(durations[i]) for i in keep]
        self.cumsum = np.cumsum(self.durations)
        print(f"total dataset duration {np.sum(self.durations) / float(self.sr)} filtered duration length: {np.sum([int(durations[i]) for i in filter]) / float(self.sr)}")

    def init_dataset(self, hps):
        self.image_clip_emb = True
        # Load list of files and starts/durations
        files = librosa.util.find_files(f'{hps.audio_files_dir}', ext=['mp3', 'opus', 'm4a', 'aac', 'wav'])
        print_all(f"Found {len(files)} files. Getting durations")
        cache = dist.get_rank() % 8 == 0 if dist.is_available() else True
        cache = True
        durations = np.array([get_duration_sec(file, cache=cache) * self.sr for file in files])  # Could be approximate
        self.clip_emb = hps.clip_emb
        self.video_clip_emb = hps.video_clip_emb
        self.file2CLIP = {}  # Initialize to avoid AttributeError
        # Note: Removed hardcoded max_duration=10.5 as it was filtering out all files
        # max_duration is already set from hps or defaults to inf in __init__
        if self.labels:
            if hps.clip_emb:
                self.CLIP_path = hps.file2CLIP
                # Load single CLIP.pickle file for image embeddings
                clip_pickle_path = os.path.join(self.CLIP_path, 'CLIP.pickle')
                if os.path.exists(clip_pickle_path):
                    with open(clip_pickle_path, 'rb') as handle:
                        CLIP_dict = pickle.load(handle)
                    self.file2CLIP = CLIP_dict.get("image", {})
                    print_all(f"Loaded {len(self.file2CLIP)} image CLIP embeddings from {clip_pickle_path}")
                else:
                    self.file2CLIP = {}
                    print_all(f"Warning: CLIP.pickle not found at {clip_pickle_path}")

        requiredTypes = None
        self.filter(files, durations, requiredTypes)

    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length//2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        offset = item * self.sample_length + shift # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f'Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}'
        index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index] # start and end of current song
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        if offset > end - self.sample_length: # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start: # Going under song
            offset = min(end - self.sample_length, offset + half_interval)  # Now should fit
        assert start <= offset <= end - self.sample_length, f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"
        offset = offset - start
        return index, offset

    def get_clip_metadata(self, fileName, test, offset=None):
        if self.image_clip_emb:
            # For image CLIP embeddings, load from pre-loaded dictionary
            if fileName in self.file2CLIP:
                clip_emb = self.file2CLIP[fileName]
                # Image CLIP embeddings can be:
                # - Single vector (512,) for one image
                # - Array of vectors (N, 512) for multiple images of same class
                if len(clip_emb.shape) == 1:
                    # Single image embedding
                    return clip_emb
                elif len(clip_emb.shape) == 2:
                    # Multiple images, use first one
                    return clip_emb[0] if clip_emb.shape[0] > 0 else clip_emb.mean(axis=0)
                else:
                    raise ValueError(f"Unexpected CLIP embedding shape: {clip_emb.shape}")
            else:
                # Fallback: try loading individual pickle file (for video case)
                pickle_path = os.path.join(self.CLIP_path, f'{fileName}.pickle')
                if os.path.exists(pickle_path):
                    with open(pickle_path, 'rb') as handle:
                        clip_emb = pickle.load(handle)[fileName]
                    frames_per_sec = 30.0
                    offset_in_sec, length_in_sec = offset / float(self.sr), self.sample_length / float(self.sr)
                    start = int(offset_in_sec * frames_per_sec)
                    end = int(start + length_in_sec*frames_per_sec)
                    sample_clip_emb = clip_emb[start:end+1]
                    mean_sample_clip_emb = np.mean(sample_clip_emb, axis=0)
                    if self.video_clip_emb:
                        return sample_clip_emb
                    else:
                        return mean_sample_clip_emb
                else:
                    raise KeyError(f"CLIP embedding not found for {fileName}")
        else:
            return self.file2CLIP[int(fileName)]


    def get_video_chunk(self, index, offset, test=False):
        filename, total_length = self.files[index], self.durations[index]
        data, sr = load_audio(filename, sr=self.sr, offset=offset, duration=self.sample_length)
        if data.ndim == 1:
            data = data[None, :]  # Add channel dimension
        elif data.shape[0] != self.channels:
            data = data[:self.channels, :]  # Keep first channels if multi-channel

        assert data.shape == (self.channels, self.sample_length), f'Expected {(self.channels, self.sample_length)}, got {data.shape}'
        if self.labels:
            if self.clip_emb:
                fileName = os.path.splitext(os.path.basename(filename))[0]
                clip_emb = self.get_clip_metadata(fileName, test, offset)
                pos = np.array([total_length, offset, self.sample_length], dtype=np.int64)
                if self.video_clip_emb:
                    pose_tiled = np.tile(pos, (clip_emb.shape[0], 1))
                    y = np.concatenate([pose_tiled, clip_emb], axis=1, dtype=np.float32)
                    return data.T, y
                else:
                    return data.T, np.concatenate([pos, clip_emb.flatten()], dtype=np.float32)
        else:
            return data.T

    def get_item(self, item, test=False):
        index, offset = self.get_index_offset(item)
        return self.get_video_chunk(index, offset, test)

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)
