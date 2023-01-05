function dot_problem(x, y)
	%epsilon = 1;
  epsilon = 1e-4;
	X = [ x.^0 x ];
	lb = [-inf 0];

	irp_temp = ir_problem(X, y, epsilon * 5);#, lb);

	figure('position',[0, 0, 800, 600]);
	ir_scatter(irp_temp);

	b_lsm = (X \ y)';
	fprintf("b1: %d, b2: %d", b_lsm(1), b_lsm(2));
	MNK_line = [b_lsm(1) + b_lsm(2) * min(x), b_lsm(1) + b_lsm(2) * max(x)];

	figure('position', [0, 0, 800, 600]);
	plot(x, y, "o");
	hold on;
	grid off;
	plot([min(x),  max(x)], MNK_line);
	saveas(gcf, "../report/lab2/dot_mse.png","png");

	## Графическое представление информационного множества
	figure('position',[0, 0, 800, 600]);
	ir_plotbeta(irp_temp);
	grid off;
	set(gca, 'fontsize', 12);
	xlabel('\beta_0');
	ylabel('\beta_1');
	title('Information set');
	saveas(gcf, "../report/lab2/dot_info_set.png","png");
endfunction
