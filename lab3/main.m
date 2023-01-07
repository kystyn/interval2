pkg load interval

x = transpose([51:150]);
%load("data1.csv")
%y = data1(51:150, 1);
load("data2.csv")
y = data2(51:150, 1);

irp_temp = interval_problem(x, y);
[px, py] = ir_plotmodelset(irp_temp, [51 150])
%[b_maxdiag, b_gravity] = parameters(x, y, irp_temp);
%joint_depth(irp_temp, b_maxdiag, b_gravity);
prediction(x, y, irp_temp, b_maxdiag, b_gravity);
%edje_points(x, y, irp_temp);
save px.mat px 
save py.mat py 