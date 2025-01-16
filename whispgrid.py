import whisper_timestamped as whisper
from whisper.tokenizer import get_tokenizer
import tgt
import json
import sv_ttk
import os
import time

import datetime
import subprocess

import argparse

import torch
import pyannote.audio

from pyannote.audio import Audio
from pyannote.core import Segment

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

SPELL_OUT_NUMBERS = False
WORD_TIER = False

MODELS = ["large-v2", "base", "small", "medium", "large", "bofenghuang/whisper-medium-french"]
LANGUAGES = ["en", "fr", "es", "de"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st = time.process_time()


embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device = device)

def transcribe_audios(args):
    start_time = time.time() 
    num_speakers = args.speakers

    initials = []
    for i in range(0, num_speakers):
        initials.append(str(i))

    def format_time(seconds):
        # Convert seconds to hours, minutes, and seconds
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    model = whisper.load_model(args.model, device=device)

    for audio_path in tqdm(args.audio_files):
        transcribe_audio(audio_path, model, args.language, int(num_speakers))

    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    formatted_time = format_time(elapsed_time)
    et = time.process_time()
    res = et - st
    print(f"Transcription Complete\nBatch transcription complete\nElapsed Time: {formatted_time} \nCPU execution time: {res} seconds")


def transcribe_audio(audio_path, model, lang, num_speakers):
    
    original_file_name, original_file_ext = os.path.splitext(os.path.basename(audio_path))

    audio = whisper.load_audio(audio_path)
    
    if SPELL_OUT_NUMBERS:
    
        tokenizer = get_tokenizer(multilingual=True)
        number_tokens = [
            i 
            for i in range(tokenizer.eot)
            if all(c in "0123456789" for c in tokenizer.decode([i]).strip())
        ]

        result = whisper.transcribe(
            model,
            audio,
            language=lang,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            trust_whisper_timestamps=False,
            suppress_tokens=[-1] + number_tokens,
        )
    
    else:
        
        result = whisper.transcribe(
            model,
            audio,
            language=lang,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            trust_whisper_timestamps=False
        )
    

    segments = result["segments"]
    previous_end_time = 0.0


    #Create TextGrid and edit it

    tg = tgt.TextGrid()

    sentences_tier = tgt.IntervalTier(start_time=0, end_time=result["segments"][-1]["end"], name="phrase")

    if WORD_TIER:
        word_tier = tgt.IntervalTier(start_time=0, end_time=result["segments"][-1]["end"], name="mot")

        #work on words before it's too late
        for segment in result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    interval = tgt.Interval(start_time=float(word["start"]), end_time=float(word["end"]), text=word["text"])
                    word_tier.add_interval(interval)

        tg.add_tier(word_tier)


    sentences_tier = tgt.IntervalTier(start_time=0, end_time=result["segments"][-1]["end"], name="phrase")

    if int(num_speakers) > 1:

        if audio_path[-3:] != 'wav':
            subprocess.call(['ffmpeg', '-i', audio_path, f'{original_file_name}.wav', '-y'])
            audio_path = f'{original_file_name}.wav'

        with contextlib.closing(wave.open(audio_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        audio_dia = Audio()

        def segment_embedding(segment):
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio_dia.crop(audio_path, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)

        embeddings = np.nan_to_num(embeddings)

        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = str(labels[i] + 1)
            try:
                speaker_initial = initials[int(segments[i]["speaker"]) - 2]
            except IndexError:
                # Handle the case where the label is out of range
                speaker_initial = "Unknown"
            #speaker_initial = initials[labels[i] + 1]
            #segments[i]["speaker"] = speaker_initial
            #segments[i]["speaker"] = str(labels[i] + 1)

            concatenated_text = f"{speaker_initial} {segments[i]['text']}"
            interval = tgt.Interval(start_time=float(segments[i]["start"]), end_time=float(segments[i]["end"]), text=concatenated_text)
            sentences_tier.add_interval(interval)

    else:
        for segment in result["segments"]:
            interval = tgt.Interval(start_time=float(segment["start"]), end_time=float(segment["end"]), text=segment["text"])
            sentences_tier.add_interval(interval)


    tg.add_tier(sentences_tier)

    input_file_name = os.path.basename(audio_path)
    output_file_name = os.path.splitext(input_file_name)[0] + ".TextGrid"
    output_path = os.path.join(os.path.dirname(audio_path), output_file_name)

    tgt.write_to_file(tg, output_path, format='short')



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio-files", type=str, nargs="+")

    parser.add_argument("--model", type=str, choices=MODELS, default=MODELS[0])
    parser.add_argument("--language", type=str, choices=LANGUAGES, default=LANGUAGES[0])

    parser.add_argument("--speakers", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    print(device)
    transcribe_audios(args)


