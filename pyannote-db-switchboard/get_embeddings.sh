#!/bin/bash

# Need to be set for each machine!
export ROOT_DIR=/home/yaniv/Projects/pyannote-audio/tutorials/speaker-embedding
# Real output of the .h5 files
OUTPUT_DIR=data/swbI_release2/preprocessed/trimmed

# Dir of the validation results of the trained model (that should be used now)
#VALIDATION_DIR=$ROOT_DIR/2+0.5/TristouNet/train/Etape.SpeakerDiarization.TV.train/validate/Etape.SpeakerDiarization.TV/

VALIDATION_DIR=$ROOT_DIR/1+0.5/TristouNet/train/LibriSpeech.SpeakerDiarization.LibriSpeechClean.train/validate/LibriSpeech.SpeakerDiarization.LibriSpeechClean

# Temp output dir
now=`date '+%Y_%m_%d__%H_%M_%S'`
EMBEDDING_TMP_OUTPUT_DIR="/tmp/.pyaudio_embeddings_results_${now}"

# The step size between windows - the duration is inherently decided by the trained model
STEP_SIZE=0.5

# Create MDTM files for the embedding extractor
echo "Generating mdtm files containing data for the extractor.."
python pyannote-db-switchboard/SwitchBoard/generate.py $OUTPUT_DIR

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
