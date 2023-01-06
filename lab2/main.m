pkg load interval

x = transpose([151:200]);
load("data1.csv")
%y = data1(151:200, 1);
load("data2.csv")
y = data2(151:200, 1);

irp_temp = interval_problem(x, y);
%[b_maxdiag, b_gravity] = parameters(x, y, irp_temp);
%joint_depth(irp_temp, b_maxdiag, b_gravity);
%prediction(x, y, irp_temp, b_maxdiag, b_gravity);
%edje_points(x, y, irp_temp);
