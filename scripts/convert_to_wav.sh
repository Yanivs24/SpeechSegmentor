#!/bin/bash

dir_path=$1


for f in ${dir_path}/*.sph
do
    sox -t sph "$f" -c 1 -r 16000 -b 16  -t wav "${f%.*}.wav"
done
