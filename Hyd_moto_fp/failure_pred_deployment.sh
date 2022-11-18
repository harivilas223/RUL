#!/bin/bash
now_s=`date`
echo "Failure pred model 1, started at $now_s">>/home/srinivas/SONA_BLW/Hyd_moto_fp/failure_pred_deployment.log

echo "##### Failure pred model  started #####">>/home/srinivas/SONA_BLW/Hyd_moto_fp/failure_pred_deployment.txt
cd /home/srinivas/SONA_BLW && virtualenv1/bin/python3 Hyd_moto_fp/failure_pred_deployment.py>>/home/srinivas/SONA_BLW/Hyd_moto_fp/failure_pred_deployment.txt

now_e=`date`
echo "Failure pred model 1, ended at $now_e">>/home/srinivas/SONA_BLW/Hyd_moto_fp/failure_pred_deployment.log


