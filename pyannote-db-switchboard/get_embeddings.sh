#!/bin/bash

# Need to be set for each machine
PYAUDIO_ROOT=/home/yaniv/Projects/pyannote-audio

# Dir of the validation results of the trained model (that should be used now)
VALIDATION_DIR=$PYAUDIO_ROOT/tutorials/speaker-embedding/2+0.5/TristouNet/train/Etape.SpeakerDiarization.TV.train/validate/Etape.SpeakerDiarization.TV/
EMBEDDING_OUTPUT_DIR=embedding_results
# The step size between windows - tha duration is inherently decided by the trained model
STEP_SIZE=0.5

rm -rf $EMBEDDING_OUTPUT_DIR/*
python `which pyannote-speaker-embedding` apply --step=$STEP_SIZE $VALIDATION_DIR/development.eer.txt SwitchBoard.SpeakerDiarization.SwitchBoardMain $EMBEDDING_OUTPUT_DIR