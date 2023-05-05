import os
import librosa


SAMPLERATE = 22050
DURATION = 30 #seconds
SAMPLES_MUSIC = SAMPLERATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    # dictionary to store data
    data = {
        "mapping":[],
        "mfcc": [],
        "labels": []
    }
    
    samples_segment = int(SAMPLES_MUSIC / num_segments)
    
    # loop through all the genre
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        #ensure that we're not at the root level
        if dirpath is not dataset_path:
            # save the semantic label 
            dirpath_components = dirpath.split("/") #genre/blues => [genre,blues]
            data['mapping'].append(dirpath_components[-1])
            
            # load audio and build mfccs
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLERATE)
                
                for s in range(num_segments):
                    start_sample = samples_segment *s 
                    finish_sample = start_sample + samples_segment
                    
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr = sr,
                                                n_fft = n_fft,
                                                n_mfcc = n_mfcc,
                                                hop_length = hop_length)
                    mfcc = mfcc.T
                    
                    
                    
                