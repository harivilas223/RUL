#!/bin/bash
now_s=`date`
echo "Failure pred model 2, started at $now_s">>/home/srinivas/SONA_BLW/Ascent_descent_fp/failure_pred_2.log

echo "##### Failure pred model  started #####">>/home/srinivas/SONA_BLW/Ascent_descent_fp/failure_pred_2.txt
cd /home/srinivas/SONA_BLW && virtualenv1/bin/python3 Ascent_descent_fp/failure_pred_2.py>>/home/srinivas/SONA_BLW/Ascent_descent_fp/failure_pred_2.txt

now_e=`date`
echo "Failure pred model 2, ended at $now_e">>/home/srinivas/SONA_BLW/Ascent_descent_fp/failure_pred_2.log



