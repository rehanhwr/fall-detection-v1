#!/bin/sh
##$ -cwd
##$ -l q_node=1
##$ -l h_rt=0:10:00
. /etc/profile.d/modules.sh
module load cuda/10.1.105 intel


# suggested default values for training: ./script.sh -d ./data/classes/ -e 5 -b 100 -v .2  -f 0
while getopts ":e:b:d:f:v:" opt; do
  case $opt in
    e) epoch="$OPTARG"
    ;;
    b) batch_size="$OPTARG"
    ;;
    d) dataset_name="$OPTARG"
    ;;
    f) feature_extraction="$OPTARG"
    ;;
    v) validation_size="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


if [ $feature_extraction -gt 0 ]
then
  python ./squeezenet.py -dn $dataset_name -e $epoch -b $batch_size -v $validation_size -f
else
  python ./squeezenet.py -dn $dataset_name -e $epoch -b $batch_size -v $validation_size
fi

