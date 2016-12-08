"cpu.py":
	Forms Protein-Keys matrix from the dataset "ProteinDataset" and finds similarity between them in a matrix form.
	'pkmat.csv' - stores Protein-Key matrix.
	'similarity_cpu.csv' - stores all pairs protein similarity measure calculated by cpu version.

"cpu.py":
	Forms Protein-Keys matrix from the dataset "ProteinDataset" and finds similarity between them in a matrix form by parallel computation.
	'pkmat.csv' - stores Protein-Key matrix.
	'similarity_gpu.csv' - stores all pairs protein similarity measure calculated by cpu version.

"result_sum.py":
        Compare the results of similarity matrices calculated by cpu and gpu versions by summimg up all the terms and chech equality between sums.
	
Programming Language: Python, pyCUDA
OS: Windows 10 amd64
Platform: Anaconda 3.5
IDE: Spyder
Graphics Card: NVIDIA Geforce GTX 960M
	       Compute Capability: 5.0

Installation Guide: https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_PyCUDA_On_Anaconda_For_Windows?lang=en