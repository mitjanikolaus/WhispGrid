srun --gres gpu:3090:1 --partition gpu python whispgrid.py --model large-v2 --lang de --audio-files ../sound/Subj*/*.wav
