# Tinycheckpoints
x1            mae=0.001835  rmse=0.004437
y1            mae=0.000732  rmse=0.001152
z1            mae=0.000228  rmse=0.000337
box_base_yaw  mae=5.359 deg rmse=17.818 deg
lid_angle     mae=4.878 deg rmse=6.455 deg
flap_angle    mae=7.490 deg rmse=10.502 deg
lid_length    mae=0.000910  rmse=0.001348

# 10000dataset checkpoints
label                  mae          rmse         extra
x1                   0.001253   0.001563  
y1                   0.000535   0.000678  
z1                   0.000180   0.000231  
box_base_yaw         0.088700   0.223445  mae_deg=5.082 rmse_deg=12.802
lid_angle            0.096657   0.119708  mae_deg=5.538 rmse_deg=6.859
flap_angle           0.069860   0.113337  mae_deg=4.003 rmse_deg=6.494
lid_length           0.000719   0.000921  

# 10000dataset checkpoints epoch 200 lr 1e-3
label                  mae          rmse         extra
x1                   0.001094   0.001386  
y1                   0.000744   0.000907  
z1                   0.000208   0.000263  
box_base_yaw         0.056358   0.224827  mae_deg=3.229 rmse_deg=12.882
lid_angle            0.078746   0.099152  mae_deg=4.512 rmse_deg=5.681
flap_angle           0.058469   0.103557  mae_deg=3.350 rmse_deg=5.933
lid_length           0.000841   0.001061  


# 10000dataset checkpoints lr 3e-4
label                  mae          rmse         extra
x1                   0.001112   0.001410  
y1                   0.000496   0.000643  
z1                   0.000146   0.000197  
box_base_yaw         0.059218   0.215150  mae_deg=3.393 rmse_deg=12.327
lid_angle            0.062554   0.078910  mae_deg=3.584 rmse_deg=4.521
flap_angle           0.058055   0.084539  mae_deg=3.326 rmse_deg=4.844
lid_length           0.000590   0.000791  

# 10000dataset checkpoints lr 3e-4 sincos
label                  mae          rmse         extra
x1                   0.001237   0.001678  
y1                   0.000612   0.000782  
z1                   0.000211   0.000268  
box_base_yaw         0.013231   0.017824  mae_deg=0.758 rmse_deg=1.021
lid_angle            0.024562   0.032280  mae_deg=1.407 rmse_deg=1.849
flap_angle           0.048503   0.086037  mae_deg=2.779 rmse_deg=4.930
lid_length           0.000845   0.001070 

# 10kdataset checkpoints lr 3e-4sincos 20 epoch
label                  mae          rmse         extra
x1                   0.001224   0.001583  
y1                   0.000518   0.000685  
z1                   0.000196   0.000251  
box_base_yaw         0.013636   0.017510  mae_deg=0.781 rmse_deg=1.003
lid_angle            0.024008   0.030790  mae_deg=1.376 rmse_deg=1.764
flap_angle           0.056843   0.094501  mae_deg=3.257 rmse_deg=5.415
lid_length           0.000783   0.001002  

# 10kdataset checkpoints lr 3e-4sincos 40 epoch
x1                   0.001080   0.001357  
y1                   0.000419   0.000555  
z1                   0.000151   0.000200  
box_base_yaw         0.011488   0.014410  mae_deg=0.658 rmse_deg=0.826
lid_angle            0.026619   0.032596  mae_deg=1.525 rmse_deg=1.868
flap_angle           0.047868   0.076229  mae_deg=2.743 rmse_deg=4.368
lid_length           0.000606   0.000801 