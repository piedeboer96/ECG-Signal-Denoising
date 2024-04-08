fid = fopen('/Users/piedeboer/Desktop/Thesis/data/nstdb/118e24.dat');

data = fread(fid, inf, 'uint16');

fclose(fid);


signalAnalyzer();
