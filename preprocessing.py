import os
import math
import json
import librosa

DATASET_PATH = "Data"
JSON_PATH = "data.json"
SAMPLERATE = 22050
DURATION = 30 #seconds
SAMPLES_MUSIC = SAMPLERATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    # dictionary to store data
    data = {
        "mapping":[], # [classical, blues] 
        "mfcc": [],   # [ [...], [...], [...]]
        "labels": []  # [ 1, 0 , 1 ]
    }
    
    samples_segment = int(SAMPLES_MUSIC / num_segments)
    mfcc_vector = math.ceil(samples_segment/hop_length)
    
    # loop through all the genre
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        #ensure that we're not at the root level
        if dirpath is not dataset_path:
            # save the semantic label 
            dirpath_components = dirpath.split("\\") #genre/blues => [genre,blues]
            data['mapping'].append(dirpath_components[-1])
            print(f'\n processing {dirpath_components[-1]}')
            # load audio and build mfccs
            for f in filenames:
                try:
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLERATE)
                    
                    for s in range(num_segments):
                        start_sample = samples_segment *s 
                        finish_sample = start_sample + samples_segment

                        mfcc = librosa.feature.mfcc(y= signal[start_sample:finish_sample],
                                                        sr = sr,
                                                        n_fft = n_fft,
                                                        n_mfcc = n_mfcc,
                                                        hop_length = hop_length)
                        mfcc = mfcc.T
                        
                        # store mfccs with only the expected length
                        if len(mfcc) == mfcc_vector:
                            data['mfcc'].append(mfcc.tolist())
                            data['labels'].append(i-1)
                            print('{}, segment: {}'.format(file_path, s ))
                except:
                    print(" aqui um erro!! ")

    with open(json_path, "w") as fp:
        json.dump( data, fp, indent=4)
        
if __name__ == "__main__":
    save_mfcc( DATASET_PATH, JSON_PATH, num_segments=10)
                        
                    
                