import csv
import json
from pprint import pprint

# this gathers the political riding result

# get valid results from json file
with open('validated_results.json') as data_file:
	data = json.load(data_file)

# create csv file where the clean data will be stored
file = csv.writer(open('ridingresults.csv', 'wb+'))

# create variables
districts = ['Districts']		# district name
totalvotes = ['Total Voters']	# total number of votes cast
candidatevotes = []				# candidate votes
count = 0						# district count

# all known parties
candidatemapping = {
	'CON': 0,
	'LIB': 1,
	'NDP': 2,
	'GRN': 3,
	'Other': 4
}

candidatemap = {
	'CON': 0,
	'LIB': 1,
	'NDP': 2,
	'GRN': 3,
	'Other': 4
}
# setting each party with it's own array to track candidate votes
for c in candidatemap:
	candidatemap[c] = [0] * 121
	# candidatemap[c][0] = c
# print candidatemap

# sort through riding data
for riding in data['Election']['Riding']:
	# restricting results to only Ontario
	if riding['RegionID'] != 6:
		continue
	# collect the district and total votes information
	districts.append(riding['RNE'].encode('utf-8'))
	totalvotes.append(riding['TotalVoters'])

	# collect the candidate information
	for can in riding['Candidate']:
		party = can['PE']
		votes = can['V']
		if party in candidatemap:
			candidatemap[party][count] = candidatemap[party][count] + votes
			print party, candidatemap[party][count]
		else:
			candidatemap['Other'][count] = candidatemap['Other'][count] + votes
			print 'other', candidatemap['Other'][count]
	count = count + 1

# ridingresult = []
# for c in candidatemap:
# 	ridingresult.append(c)

ridingwinners = []
for c in range(0, len(candidatemap['CON'])): # 0 to 5
	maxVotes = candidatemap['CON'][c]
	maxType = 'CON'
	for canType in candidatemap:
		print canType, candidatemap[canType][c]
		if(maxVotes < candidatemap[canType][c]):
			maxVotes = candidatemap[canType][c]
			maxType = canType
		# print canType, maxVotes, maxType, candidatemap[canType][c]
	print maxType
	ridingwinners.append(candidatemapping[maxType])
# print candidatemap, len(candidatemap), len(candidatemap['CON']), count

# this gathers the demographic information
with open('ridingresultsontario.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	
	NUM_OF_DISTRICTS = 121
	NUM_OF_CHARACTERISTICS = 215

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
		if row['FED_Name'] not in districts or row['FED_Name'] != current_district:
			if district_count != -1:
				dataset[district_count].append(float(ridingwinners[district_count]))
			districts.append(row['FED_Name'].strip())
			current_district = row['FED_Name']
			district_count = district_count + 1
			characteristic_count = 0
			characteristics = []
			total_population = row['Total']

		if row['Characteristic'].strip() not in characteristics:
			if row['Characteristic'][0:5] != 'Total':
				if row['Characteristic'][2] != ' ':
					characteristics.append(row['Characteristic'].strip())
					row['Total']
					dataset[district_count][characteristic_count] = float(row['Total'])
					characteristic_count = characteristic_count + 1

	# write to the file
	with open('alldata2015.csv', 'wb') as output:
		writer = csv.writer(output)
		chararray = [characteristics]
		writer.writerows(chararray)
		for row in dataset:
			temparray = [row]
			writer.writerows(temparray)
			# print row