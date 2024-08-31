#!/bin/bash

datasets=("Calibration" "Dataset1" "Dataset2" "Occluded" "Overlapping" "PCB" "Prasad")

for dataset in "${datasets[@]}"
do
  echo "Running experiment on $dataset"
  ./Demo/EllipseDetector -D $dataset -M 2
  echo "Experiment on $dataset completed"
  echo "------------------------"
done

echo "All experiments completed"
