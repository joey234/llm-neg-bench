import pandas as pd
import sys, getopt
import numpy as np

def load_result_file(file):
	df = pd.read_csv(file)
	print(df.head())

	whr1 = []
	whr3 = []
	whr5 = []
	for i, row in df.iterrows():
		whr1.append(float(row['WHR@1'].split(',')[0].replace('tensor(','')))
		whr3.append(float(row['WHR@3'].split(',')[0].replace('tensor(','')))
		whr5.append(float(row['WHR@5'].split(',')[0].replace('tensor(','')))
	whr1 = np.nan_to_num(whr1)
	whr3 = np.nan_to_num(whr3)
	whr5 = np.nan_to_num(whr5)


	print(np.mean(whr1))
	print(np.mean(whr3))
	print(np.mean(whr5))



def main(argv):
	inputfile = ''
	outputfile = ''
	try:
		opts, args = getopt.getopt(argv,"hi:",["ifile="])
	except getopt.GetoptError:
		print("test.py -i <inputfile>")
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('test.py -i <inputfile>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
	outputfile = inputfile.split('.')[0]+'_avg.csv'
	# print('Input file is "', inputfile)
	# print('Output file is "', outputfile)
	

	load_result_file(inputfile)

if __name__=='__main__':
	
	main(sys.argv[1:])

