#!/bin/bash
# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
#     bash download_dataset.sh ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     bash download_dataset.sh WaterDrop /tmp/

# bash download-parse-dataset.sh Goop datasets/
set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/${DATASET_NAME}"

BASE_URL="https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/${DATASET_NAME}/"

mkdir -p ${OUTPUT_DIR}
for file in metadata.json train.tfrecord valid.tfrecord test.tfrecord
do
wget -O "${OUTPUT_DIR}/${file}" "${BASE_URL}${file}"
done

ACCOUNT=brandonh
N=8
GPU=p100:1
REQ_TIME=1:00:00
MEM=32G
# salloc --account=def-$ACCOUNT --cpus-per-task=$N --gres=gpu:$GPU --mem=$MEM --time=$REQ_TIME
salloc --account=def-$ACCOUNT --cpus-per-task=$N  --mem=$MEM --time=$REQ_TIME


module purge
module load python
source venv/bin/activate

python convert_tfrecord/convert_tfrecord.py datasets/${DATASET_NAME}