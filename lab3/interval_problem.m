function irp_temp = interval_problem(x, y)
	m = length(x)
	eps = ones(m, 1);
  eps = eps*1e-4;

	C = zeros(m + 2, 1);
	for i = 1:m
		C(i, 1) = 1;
	endfor
	%display(C);

	A = zeros(2 * m, m + 2);
	for i = 1:m
		A(i, i) = -eps(i);
		A(i + m, i) = -eps(i);

		A(i, m + 1) = 1;
		A(i + m, m + 1) = -1;

		A(i, m + 2) = x(i);
		A(i + m, m + 2) = -x(i);
	endfor
	%display(A);

	B = zeros(2 * m, 1);
	for i = 1:m
		B(i, 1) = y(i);
		B(i + m, 1) = -y(i);
	endfor
	%display(B);

	lb = zeros(1, m + 2);
	for i = 1:m
		lb(i) = 1;
	endfor
  lb(m + 1) = -0;
	lb(m + 2) = -0;
	%display(lb);

	ctype = "";
	for i = 1:2 * m
		ctype(i) = 'U';
	endfor
	%display(ctype);

	vartype = "";
	for i = 1:m + 2
		vartype(i) = 'C';
	endfor
	%display(vartype);
	sense = 1;

	[w, fw, errcode, lambda] = glpk(C,A,B,lb,[],ctype,vartype,sense);
  
  save w.mat w
  
  if errcode > 0
    return
  endif

  m1 = length(w)
	scale = max(w(1:m));
	for i = 1:m
		eps(i) = eps(i) * 1; %KYSTYN scale
	end

	X = [ x.^0 x ];
	lb = [-inf -inf]; % -inf 0
	irp_temp = ir_problem(X, y, eps, lb);

	display(w);
	## График интервальных измерений
	figure('position', [0, 0, 800, 600]);
	ir_scatter(irp_temp, 'bo');
	set(gca, 'fontsize', 12);
	grid on;
	saveas(gcf, "../report/lab2/interval_problem.png","png");

	%figure('position', [0, 0, 800, 600]);
	%ir_plotbeta(irp_temp);
	%grid on;
	%set(gca, 'fontsize', 12);
	%xlabel('\beta_0')
	%ylabel('\beta_1');
	%title('Information set');
	%saveas(gcf, "../report/images/interval_info_set.eps","epsc");
endfunction
