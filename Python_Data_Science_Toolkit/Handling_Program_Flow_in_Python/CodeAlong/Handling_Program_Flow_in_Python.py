import json

with open('ipl_match.json') as f:
    data = json.load(f)
print(data)

print("#" * 50)
print("Can you find how many deliveries were faced by batsman SC Ganguly ? ")
print("#" * 50)
deliveries_in_1st_innings =  data.get('innings')[0].get('1st innings').get('deliveries')
ganguly = 0
for i in range(0, len(deliveries_in_1st_innings)):
    for key in deliveries_in_1st_innings[i].keys():
        if deliveries_in_1st_innings[i].get(key).get('batsman') == "SC Ganguly":
            ganguly += 1

print(ganguly)

print("#" * 50)
print("Who was man of the match and how many runs did he scored  ? ")
print("#" * 50)
man_of_match = data.get('info').get('player_of_match')
print(man_of_match)
deliveries_in_1st_innings =  data.get('innings')[0].get('1st innings').get('deliveries')
deliveries=  len(deliveries_in_1st_innings)
runs = []
for i in range(0, deliveries):
    for key in deliveries_in_1st_innings[i].keys():
        if deliveries_in_1st_innings[i].get(key).get('batsman') == 'BB McCullum':
            run = deliveries_in_1st_innings[i].get(key).get('runs').get('batsman')
            runs.append(run)
print(sum(runs))

print("#" * 50)
print("Which batsman played in the first inning ? ")
print("#" * 50)
batsmen = []

for i in range(0, deliveries):
    for key in deliveries_in_1st_innings[i].keys():
        batsman = deliveries_in_1st_innings[i].get(key).get('batsman')
        batsmen.append(batsman)

res = []
for i in batsmen:
    if i not in res:
        res.append(i)
print(res)

print("#" * 50)
print("Which batsman played in the first inning ? ")
print("#" * 50)

batsman_out = []
deliveries_in_2nd_innings = data.get('innings')[1].get('2nd innings').get('deliveries')
deliveries_2nd = len(deliveries_in_2nd_innings)
key_2 = None

for i in range(0, deliveries_2nd):
    for key_2 in deliveries_in_2nd_innings[i].keys():
        if 'wicket' in deliveries_in_2nd_innings[i].get(key_2):
            if deliveries_in_2nd_innings[i].get(key_2).get('wicket').get(
                    'kind') == 'bowled':
                bowled = deliveries_in_2nd_innings[i].get(key_2).get('batsman')
                batsman_out.append(bowled)

print(batsman_out)

print("#" * 50)
print("How many more extras (wides, legbyes, etc) were bowled in the second innings as compared to the first inning ")
print("#" * 50)
extras_1 = []
extras_2 = []
deliveries_2nd = len(data.get('innings')[1].get('2nd innings').get('deliveries'))

for i in range(0, deliveries):
    for key in deliveries_in_1st_innings[i].keys():
        extra_1 = deliveries_in_1st_innings[i].get(key).get('runs').get('extras')
        extras_1.append(extra_1)

for i in range(0, deliveries_2nd):
    for key_2 in deliveries_in_2nd_innings[i].keys():
        extra_2 = deliveries_in_2nd_innings[i].get(key_2).get('runs').get('extras')
        extras_2.append(extra_2)

print(sum(extras_1))
print(sum(extras_2))

extra_extras = sum(extras_2) - sum(extras_1)
print(extra_extras)

print("#" * 50)
print("How many more extras (wides, legbyes, etc) were bowled in the second innings as compared to the first inning ")
print("#" * 50)