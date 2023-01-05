function joint_depth(irp_temp, b_maxdiag, b_gravity)
	## Графическое представление коридора совместных зависимостей для модели y = beta0 + beta1 * x
	figure('position',[0, 0, 800, 600]);
  xlimits = [1 200];
	ir_plotmodelset(irp_temp, xlimits);     # коридор совместных зависимостей
	hold on;
	ir_scatter(irp_temp,'bo');              # интервальные измерения
	ir_plotline(b_maxdiag, xlimits, 'r-');   # зависимость с параметрами, оцененными как центр наибольшей диагонали ИМ
	ir_plotline(b_gravity, xlim, 'b--');     # зависимость с параметрами, оцененными как центр тяжести ИМ
	grid on;
	set(gca, 'fontsize', 12);
	saveas(gcf, "../report/lab2/joint_depth.png","png");
endfunction
