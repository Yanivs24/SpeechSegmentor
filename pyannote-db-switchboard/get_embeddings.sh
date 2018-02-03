#!/bin/bash

# Need to be set for each machine!
PYAUDIO_ROOT=/home/yaniv/Projects/pyannote-audio

# Dir of the validation results of the trained model (that should be used now)
VALIDATION_DIR=$PYAUDIO_ROOT/tutorials/speaker-embedding/2+0.5/TristouNet/train/Etape.SpeakerDiarization.TV.train/validate/Etape.SpeakerDiarization.TV/

# Temp output dir
now=`date '+%Y_%m_%d__%H_%M_%S'`
EMBEDDING_TMP_OUTPUT_DIR="/tmp/.pyaudio_embeddings_results_${now}"

# Real output of the .h5 files
OUTPUT_DIR=data/swbI_release2/preprocessed/trimmed

# The step size between windows - tha duration is inherently decided by the trained model
STEP_SIZE=0.25

# Apply the network
echo "Extracting embeddings to ${EMBEDDING_TMP_OUTPUT_DIR}"
python `which pyannote-speaker-embedding` apply --step=$STEP_SIZE $VALIDATION_DIR/development.eer.txt SwitchBoard.SpeakerDiarization.SwitchBoardMain $EMBEDDING_TMP_OUTPUT_DIR

# Delete old h5 files
echo "Removing old h5 files from ${OUTPUT_DIR}"
rm -f $OUTPUT_DIR/*.h5

# Move the new files
echo "Moving new h5 files to ${OUTPUT_DIR}"
mv $EMBEDDING_TMP_OUTPUT_DIR/SwitchBoard/* $OUTPUT_DIR/

# Remove temp dir
echo "Removing ${EMBEDDING_TMP_OUTPUT_DIR}"
rm -rf $EMBEDDING_TMP_OUTPUT_DIR