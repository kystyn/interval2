pkg load interval

x = transpose([1:200]);
load("data1.csv")
y = data1(:, 1);
%load("data2.csv")
%y = data2(:, 1);

%draw_graph(x, y, "n", "mV", "", 1, "all_data.eps");
%print -djpg all_data.jpg
%draw_graph(x, y, "time", "value", "", 1, "filtered_data.eps");
%draw_graph(x, y, "time", "value", "d", 1, "selected_data.eps");

%dot_problem(x, y);

irp_temp = interval_problem(x, y);
[b_maxdiag, b_gravity] = parameters(x, y, irp_temp);
joint_depth(irp_temp, b_maxdiag, b_gravity);
%prediction(x, y, irp_temp, b_maxdiag, b_gravity);
%edje_points(x, y, irp_temp);
