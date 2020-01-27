#!/bin/sh
##$ -cwd
##$ -l q_node=1
##$ -l h_rt=0:10:00
. /etc/profile.d/modules.sh
module load cuda/10.1.105 intel


# suggested default values for training: ./script_cnf.sh -d "/gs/hs0/tga-isshiki-lab/rehan/dataset/" -b 20 -v .3  -f 0 -l './saved_model/batch294_epoch0_saved_model.pth'
while getopts ":e:b:d:f:v:c:r:l:" opt; do
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
    c) class_size="$OPTARG"
    ;;
    r) resume_training="$OPTARG"
    ;;
    l) load_path="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


if [ $feature_extraction -gt 0 ]
then
  python ./confusion_matrix.py -dn $dataset_name -e $epoch -b $batch_size -v $validation_size -c $class_size -f
else
  if [ $resume_training -gt 0 ]
  then
    python ./confusion_matrix.py -dn $dataset_name -e $epoch -b $batch_size -v $validation_size -c $class_size -r $resume_training -l $load_path
  else
    python ./confusion_matrix.py -dn $dataset_name -e $epoch -b $batch_size -v $validation_size -c $class_size
  fi
fi

