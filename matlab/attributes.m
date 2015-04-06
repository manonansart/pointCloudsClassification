
% Expected path format : 'foo/'

function att = attributes(fpath)
	att = [];
	% Loop over all pcd files in fpath
	files = dir(strcat(fpath, '*.pcd'));
	for file = files'
	    data = loadpcd(strcat(fpath, file.name));

	    intensity = data(4, :)';

	    S = cov(data(1:3, :)');
	    vp = eig(S);
	    vpXData = vp(1);
	    vpYData = vp(2);
	    vpZData = vp(3);

		att = [att ;
			min(intensity) max(intensity) mean(intensity) var(intensity) vpXData vpYData vpZData];
	end

	% Save the attributes to csv format
	csvwrite(strcat(fpath,'attributes.csv'), att);

end