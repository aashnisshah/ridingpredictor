import urllib2
import csv
from bs4 import BeautifulSoup

url_election_ca = []
base_url = 'http://www.elections.ca/Scripts/VIS/HistoricalResults/'
base_end_url = '_e.html'

for i in range(1, 106):
	url_value = 35000 + i
	new_url = base_url + str(url_value) + base_end_url
	url_election_ca.append(new_url)

for url_dis in url_election_ca:
	response = urllib2.urlopen(url_dis)
	html = response.read()

	soup = BeautifulSoup(html, 'html.parser')

	candidatemapping = {
		'Conservative Party of Canada': 0,
		'Liberal Party of Canada': 1,
		'New Democratic Party': 2,
		'GRN': 3,
		'Other': 4
	}

	output_data = []
	year_result = []
	year_count = 0

	district = soup.find_all('caption')[0].text

	years = []
	for year in soup.find_all('h4'):
		years.append(year.text[-5: -1])

	for row in soup.find_all('tr'):
		if 'Total' in row.text:
			year_result = [district, years[year_count], year_result]
			output_data.append(year_result)
			year_count = year_count + 1
			year_result = []

		if 'Candidate' not in row.text and 'Total' not in row.text:
			row_data = row.text.split('\n')
			row_result = [row_data[2], row_data[4]]
			year_result.append(row_result)

	data_for_csv = []
	for data in output_data:
		if data[1] == '2006':
			# print data[0], data[1]
			maxVotes = 0
			maxPos = 0
			for results in data[2]:
				# print results[0], results[1]
				if float(results[1]) > float(maxVotes):
					maxVotes = results[1]
					maxPos = results[0]
			# print '>>>> ', data[0], ': ', maxVotes, results[1], maxPos
			print data[0], ',', candidatemapping[maxPos]
			if 'Ottawa--Orl' in data[0]:
				data[0] = 'Ottawa--Orleans (Ontario)'
			data_for_csv.append([data[0], candidatemapping[maxPos]])
			# print '\n'
	# print data_for_csv

	with open('alldatavotes.csv', 'wb') as output:
		writer = csv.writer(output)
		# writer.writerows(chararray)
		for row in data_for_csv:
			temparray = [row]
			# print '>>>>',row
			writer.writerows([row])
			# print row

	# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>"