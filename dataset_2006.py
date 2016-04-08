import csv
import json
from pprint import pprint

# this gathers the political riding result

# # get valid results from json file
# with open('validated_results.json') as data_file:
# 	data = json.load(data_file)

# # create csv file where the clean data will be stored
# file = csv.writer(open('ridingresults.csv', 'wb+'))

# # create variables
# districts = ['Districts']		# district name
# totalvotes = ['Total Voters']	# total number of votes cast
# candidatevotes = []				# candidate votes
# count = 0						# district count

# # all known parties
# candidatemapping = {
# 	'CON': 0,
# 	'LIB': 1,
# 	'NDP': 2,
# 	'GRN': 3,
# 	'Other': 4
# }

# candidatemap = {
# 	'CON': 0,
# 	'LIB': 1,
# 	'NDP': 2,
# 	'GRN': 3,
# 	'Other': 4
# }
# # setting each party with it's own array to track candidate votes
# for c in candidatemap:
# 	candidatemap[c] = [0] * 121
# 	# candidatemap[c][0] = c
# # print candidatemap

# # sort through riding data
# for riding in data['Election']['Riding']:
# 	# restricting results to only Ontario
# 	if riding['RegionID'] != 6:
# 		continue
# 	# collect the district and total votes information
# 	districts.append(riding['RNE'].encode('utf-8'))
# 	totalvotes.append(riding['TotalVoters'])

# 	# collect the candidate information
# 	for can in riding['Candidate']:
# 		party = can['PE']
# 		votes = can['V']
# 		if party in candidatemap:
# 			candidatemap[party][count] = candidatemap[party][count] + votes
# 		else:
# 			candidatemap['Other'][count] = candidatemap['Other'][count] + votes
# 	count = count + 1

# ridingresult = []
# for c in candidatemap:
# 	ridingresult.append(c)

# ridingwinners = []
# for c in range(0, len(candidatemap['CON'])):
# 	maxVotes = candidatemap['CON'][c]
# 	maxType = 'CON'
# 	for canType in candidatemap:
# 		if(maxVotes < candidatemap[canType][c]):
# 			maxVotes = candidatemap[canType][c]
# 			maxType = canType
# 	ridingwinners.append(candidatemapping[maxType])




# with open('ridingresultsontario_2006.csv') as csvfile:
# 	reader = csv.DictReader(csvfile)

# 	count = 0
# 	prev_csd = ''
# 	other_count = 0
# 	characteristics = []
# 	characteristic_count = 0

# 	for row in reader:
# 		# print row['CSD_Name']
# 		if row['CSD_Name'] != prev_csd:
# 			print '>>>>>>>>>>>>>>>>>>>>>>>>'
# 			print count, other_count, prev_csd, row['Characteristic']
# 			count = count + 1
# 			other_count = 0
# 			prev_csd = row['CSD_Name']
# 		else:
# 			other_count = other_count + 1

# 		if count == 584:
# 			print row['Characteristic']

# 		if count == 584:
# 			if row['Characteristic'].strip() not in characteristics:
# 				if row['Characteristic'][0:5] != 'Total':
# 					if row['Total'] == '':
# 						print row['Characteristic'], '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'

# 	print count, other_count




# this gathers the demographic information
with open('ridingresultsontario_2006.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	
	NUM_OF_DISTRICTS = 585
	NUM_OF_CHARACTERISTICS = 220

	districts = []
	current_district = ''
	district_count = -1
	total_population = 0

	# Create dataset based on number of districts and characteristics per district
	dataset = [0] * NUM_OF_DISTRICTS
	for num in range(0, NUM_OF_DISTRICTS):
		dataset[num] = [0] * NUM_OF_CHARACTERISTICS

	# aggregate the totals for each district
	characteristics = []
	characteristic_count = 0
	for row in reader:
		if row['CSD_Name'] not in districts or row['CSD_Name'] != current_district:
			if district_count != -1:
				# dataset[district_count].append(float(ridingwinners[district_count]))
				dataset[district_count].append(1)
			districts.append(row['CSD_Name'].strip())
			current_district = row['CSD_Name']
			district_count = district_count + 1
			characteristic_count = 0
			characteristics = []
			total_population = row['Total']
			if total_population == '':
				# print row['CSD_Name']
				total_population = 0
			print row['CSD_Name']

		if row['Characteristic'].strip() not in characteristics:
			if row['Characteristic'][0:5] != 'Total':
				if row['Total'] == '':
					row['Total'] = 0
				characteristics.append(row['Characteristic'].strip())
				if float(total_population) == 0:
					new_dataset_percentage = 0
				else:
					new_dataset_percentage = float(row['Total']) / float(total_population)
				dataset[district_count][characteristic_count] = new_dataset_percentage
				characteristic_count = characteristic_count + 1

	# write to the file
	with open('alldata_2006.csv', 'wb') as output:
		writer = csv.writer(output)
		chararray = [characteristics]
		writer.writerows(chararray)
		for row in dataset:
			temparray = [row]
			writer.writerows(temparray)
			# print row