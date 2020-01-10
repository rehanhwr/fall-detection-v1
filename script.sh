#!/bin/sh
##$ -cwd
##$ -l q_node=1
##$ -l h_rt=0:10:00
. /etc/profile.d/modules.sh
module load cuda/10.1.105 intel
python ./squeeze_net2.py -dn data -e 15