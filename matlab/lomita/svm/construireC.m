function C = construireC(Yapp, c)
	C = zeros(size(Yapp, 1), 1);
	
	Nmoins = length(find(Yapp == -1));
	Nplus = length(find(Yapp == 1));
	N = (Nmoins + Nplus);
	gammaPlus = Nplus / N;
	gammaMoins = Nmoins / N;
	
	indp = find(Yapp == 1);
	C(indp) = c / gammaPlus;
	
	indn = find(Yapp == -1);
	C(indn) = c / gammaMoins;
end