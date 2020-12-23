# CyberSecurityFinalProject
zw593, xg886, cwc478, hw2413 


## How to run
For each eval.py, simply type:  
python eval.py [query data] 
(query data should be .h5 files.) 

E.g for badnet1  
python eva1.py query_data_path

- eval1.py for sunglasses_bd_net.h5
- eval2.py for anonymous_1_bd_net.h5
- eval3.py for anonymous_2_bd_net.h5
- eval4.py for multi_trigger_target_bd_net.h5

#
Test Data
Model
Accuracy(Original)
Accuracy(defence)
1
clean_test_data
sunglasses_bd_net
97.779%
90.569%
2
clean_test_data
anonymous_1_bd_net
97.186%
91.567%
3
clean_test_data
anonymous_2_bd_net
95.966%
90.273%
4
clean_test_data
multi_trigger_multi_target_bd_net
96.009%
88.426%
